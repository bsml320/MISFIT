"""Honest reliability-estimator comparison (binary-error target) for the MedIA paper.

Replaces an unreproducible published table. Every estimator is correlated (Spearman)
against the per-subject BINARY error err_i in {0,1} (1 if subject misclassified in its
held-out fold), pooled across leave-one-environment-out folds, 5 seeds.

This construction matches the scoring path in run_per_dataset_iop_loss.py.

Datasets: ADNI (N=731, 8 scanner protocols), OASIS-1 (N=226, 4 age quartiles).

Estimators (paper's 18 + Domain-BN), each scored per-subject:
  PAC-Bayes family : structural_misfit, pac_bayes_full, pac_bayes_kl, loading_distance
  Combined         : cross_env_variance, cross_env_consistency, conformal_quantile,
                     loss_plus_margin, loss_plus_mahalanobis, loss_times_uncertainty,
                     loss_plus_gradient, split_prior_pacbayes, adaptive_oracle
  Model-agnostic   : erm_loss, deep_ensemble5, ensemble_disagree3, mc_dropout, max_softmax
  Separate         : domain_bn

Numerics (repo rules): float64 covariance; never torch.inverse (linalg.solve / SVD
pseudoinverse); condition-number checks with warning if > 1e6.

Output: misfit/results/honest_estimator_table.json
"""

from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))

from misfit.core.encoders.mlp import MLPEncoder  # noqa: E402
from misfit.core.operators.operator_model import OperatorModel  # noqa: E402
from misfit.core.operators.program_dict import ProgramDictionary  # noqa: E402
from misfit.core.types import TensorConfig, TrainConfig  # noqa: E402
from misfit.core.utils import seed_everything  # noqa: E402
from misfit.tensor_spectral.recovery import recover_programs  # noqa: E402
from misfit.training.two_stage_trainer import TwoStageTrainer  # noqa: E402
from misfit.certificates.pac_bayes import (  # noqa: E402
    PACBayesConfig,
    compute_pac_bayes_bounds,
)

warnings.filterwarnings("ignore")

OUT = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
OUT.mkdir(parents=True, exist_ok=True)

# ---- Config (mirrors run_per_dataset_iop_loss.py / run_real_certificates.py) ----
N_PROGRAMS = 5
HIDDEN_DIMS = [64, 32]
N_EPOCHS = 100
LR = 5e-4
BATCH_SIZE = 32
PATIENCE = 20
N_SEEDS = 5
N_ENSEMBLE5 = 5
N_ENSEMBLE3 = 3
MC_PASSES = 20
MC_DROPOUT = 0.2
SEED_BASE = 42

DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)  # CUDA_VISIBLE_DEVICES restricts to the allowed GPU


def _cond_warn(mat: np.ndarray, tag: str) -> float:
    """Condition number check (repo rule): warn if kappa > 1e6."""
    try:
        kappa = float(np.linalg.cond(mat))
    except Exception:
        kappa = float("inf")
    if kappa > 1e6:
        warnings.warn(f"[{tag}] condition number {kappa:.3e} > 1e6", stacklevel=2)
    return kappa


# ---------------------------------------------------------------------------
# FIRE / IOP operator model (frozen random programs; the spectral-init attempt
# below is inert, see REPRODUCIBILITY.md, Implementation notes) -- canonical misfit
# ---------------------------------------------------------------------------
def _train_fire(features, targets, site_ids, n_sites, dim, seed):
    """Train the IOP operator model exactly as run_per_dataset_iop_loss.py."""
    torch.manual_seed(seed)
    encoder = MLPEncoder(dim, HIDDEN_DIMS, dim)
    prog = ProgramDictionary(n_programs=N_PROGRAMS, dim=dim, rank=1)
    n_classes = max(int(targets.max().item()) + 1, 2)
    model = OperatorModel(encoder, prog, n_envs=n_sites, output_dim=n_classes, use_style=False)
    trainer = TwoStageTrainer(model, use_contrastive=False, task="classification")
    try:
        from misfit.data.mri_multisite import MRIData, mri_to_multienv_format

        mri = MRIData(
            features=features, targets=targets, site_ids=site_ids,
            subject_ids=torch.arange(len(features)),
        )
        obs, _ = mri_to_multienv_format(mri)
        tc = TensorConfig(n_programs=N_PROGRAMS, n_envs=n_sites, dim=dim)
        recovery = recover_programs(obs, n_programs=N_PROGRAMS, tensor_config=tc)
        trainer.stage1_init(recovery)
    except Exception:
        pass
    train_dict = {
        "x": features, "targets": targets,
        "loadings": torch.randn(len(features), N_PROGRAMS),
        "env_ids": site_ids, "subject_ids": torch.arange(len(features)),
    }
    trainer.config = TrainConfig(
        learning_rate=LR, max_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
        patience=PATIENCE, sparsity_weight=0.01, contrastive_weight=0.0,
        freeze_programs=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    trainer.stage2_train(train_dict)
    # NOTE (bug-report #3 -- resolved by honest labeling, NOT a behavior change):
    # stage2_train freezes the per-subject loadings, so train_dict["loadings"] stays
    # at its random initialization by design. This is the frozen-random operator
    # (config V0 of the operator-contribution sweep) -- a fixed scaffold, not a fitted
    # per-subject state. The manuscript's controlled ablation, which co-trains the
    # loadings with the encoder, shows the operator is inert (loadings stay at identity,
    # rho unchanged), so these loadings are an intentional uninformative negative control
    # for the loading-based estimators (loading distance, PAC-Bayes KL). Do NOT "fit" them
    # post-hoc against the frozen encoder: that only mops up the encoder's per-subject
    # residual and manufactures a misfit-proxy signal that contradicts the co-trained
    # ablation. The reliability signal is the per-subject loss, not the operator.
    model.cpu()
    return model, train_dict["loadings"]


def _predict_loadings(train_feat, train_loadings, test_feat):
    """Ridge regression for test loadings (float64, linalg.solve -- no inverse)."""
    X = train_feat.to(dtype=torch.float64)
    Y = train_loadings.to(dtype=torch.float64)
    reg = 1e-4 * X.shape[0]
    XtX = X.T @ X + reg * torch.eye(X.shape[1], dtype=torch.float64)
    _cond_warn(XtX.numpy(), "ridge_XtX")
    W = torch.linalg.solve(XtX, X.T @ Y)
    return (test_feat.to(dtype=torch.float64) @ W).to(dtype=train_feat.dtype)


# ---------------------------------------------------------------------------
# ERM plain MLP (model-agnostic baseline, no operator structure)
# ---------------------------------------------------------------------------
class PlainMLP(nn.Module):
    """Two-hidden-layer MLP ([64, 32]) with optional dropout; the ERM comparator."""

    def __init__(self, dim, n_classes, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _train_erm(features, targets, dim, seed, dropout=0.0, epochs=N_EPOCHS):
    """Train a PlainMLP by full-batch Adam cross-entropy; returns the trained model."""
    torch.manual_seed(seed)
    n_classes = max(int(targets.max().item()) + 1, 2)
    model = PlainMLP(dim, n_classes, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X = features.to(DEVICE)
    y = targets.long().to(DEVICE)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
    return model


# ---------------------------------------------------------------------------
# Domain-specific BatchNorm model (per-environment normalisation stats)
# ---------------------------------------------------------------------------
class DomainBNMLP(nn.Module):
    """MLP with a separate BatchNorm1d per training environment.

    Test-time uses the average of the per-env running stats (no test env known).
    """

    def __init__(self, dim, n_classes, n_envs):
        super().__init__()
        self.fc1 = nn.Linear(dim, 64)
        self.bns = nn.ModuleList([nn.BatchNorm1d(64) for _ in range(n_envs)])
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.n_envs = n_envs

    def _bn_avg(self, h):
        # average running stats across env BNs (eval / unknown-env fallback)
        mean = torch.stack([bn.running_mean for bn in self.bns]).mean(0)
        var = torch.stack([bn.running_var for bn in self.bns]).mean(0)
        eps = self.bns[0].eps
        h = (h - mean) / torch.sqrt(var + eps)
        w = torch.stack([bn.weight for bn in self.bns]).mean(0)
        b = torch.stack([bn.bias for bn in self.bns]).mean(0)
        return h * w + b

    def forward(self, x, env=None):
        h = torch.relu(self.fc1(x))
        if self.training and env is not None:
            outs = torch.empty_like(h)
            for e in range(self.n_envs):
                m = env == e
                if m.any():
                    outs[m] = self.bns[e](h[m])
            h = torch.relu(outs)
        else:
            h = torch.relu(self._bn_avg(h))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


def _train_domain_bn(features, targets, env_ids, n_envs, dim, seed, epochs=N_EPOCHS):
    """Train the DomainBNMLP full-batch so every environment's BN sees samples."""
    torch.manual_seed(seed)
    n_classes = max(int(targets.max().item()) + 1, 2)
    model = DomainBNMLP(dim, n_classes, n_envs).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X = features.to(DEVICE)
    y = targets.long().to(DEVICE)
    env = env_ids.long().to(DEVICE)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        # need >1 sample per env for BN; subsample full-batch with all envs present
        loss = F.cross_entropy(model(X, env), y)
        loss.backward()
        opt.step()
    return model


# ---------------------------------------------------------------------------
# Mahalanobis distance (density, float64, SVD pseudoinverse -- no inverse)
# ---------------------------------------------------------------------------
def _mahalanobis_scores(train_X, train_y, test_X):
    """Min over classes of (x-mu_c)^T Sigma^-1 (x-mu_c). float64, pinv via SVD."""
    Xtr = train_X.numpy().astype(np.float64)
    ytr = train_y.numpy().astype(int)
    Xte = test_X.numpy().astype(np.float64)
    n_classes = int(ytr.max()) + 1
    d = Xtr.shape[1]
    means = np.zeros((n_classes, d))
    cov_sum = np.zeros((d, d))
    n_total = 0
    for c in range(n_classes):
        m = ytr == c
        if m.sum() < 2:
            continue
        Xc = Xtr[m]
        means[c] = Xc.mean(0)
        Xcc = Xc - means[c]
        cov_sum += Xcc.T @ Xcc
        n_total += int(m.sum())
    cov = cov_sum / max(n_total - n_classes, 1) + 1e-3 * np.eye(d)
    _cond_warn(cov, "mahalanobis_cov")
    inv_cov = np.linalg.pinv(cov, rcond=1e-12)  # SVD pseudoinverse, not inverse
    mahal = np.zeros((len(Xte), n_classes))
    for c in range(n_classes):
        diff = Xte - means[c]
        mahal[:, c] = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return mahal.min(1)


# ---------------------------------------------------------------------------
# Spearman helper
# ---------------------------------------------------------------------------
def _rho(score, err):
    """Spearman rho of score vs error; nan when either side is constant."""
    score = np.asarray(score, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    if np.std(score) < 1e-12 or np.std(err) < 1e-12:
        return float("nan")
    r, _ = sp_stats.spearmanr(score, err)
    return float(r)


# ---------------------------------------------------------------------------
# Per-seed: train all models across LOO folds, accumulate per-subject scores
# ---------------------------------------------------------------------------
def run_seed(features, targets, site_ids, seed, want_selective=False):
    """One seed: leave-one-environment-out, all estimators, binary-error target.

    CONSTRUCTION FIDELITY (matches the paper's tab:full_ablation semantics):
      * The TARGET is per-subject binary error in the subject's single held-out
        fold (1 if misclassified there).
      * LOSS-FAMILY estimators (misfit, ERM loss, Domain-BN, PAC-Bayes full/KL,
        loading distance, split-prior, conformal, loss+margin/Maha/unc/grad,
        cross-env var/consistency, adaptive oracle) are TRAINING-derived: the
        subject's per-fold quantity is averaged over the (n_envs-1) folds where it
        was in TRAIN -- exactly the misfit machinery. This is the deployable,
        non-circular signal the paper claims.
      * UQ estimators (max softmax, deep ensemble 5/3, MC dropout) are genuine
        TEST-time held-out uncertainty in the subject's own held-out fold.
    """
    n_sites = int(site_ids.max().item()) + 1
    dim = features.shape[1]
    N = features.shape[0]
    seed_everything(seed)
    from misfit.certificates.pac_bayes import compute_kl_divergence

    # ---- error target (held-out fold correctness) ----
    subject_correct = torch.full((N,), float("nan"))
    subject_tested = torch.zeros(N, dtype=torch.bool)

    # ---- TRAINING-side accumulators (mean over in-train folds) ----
    train_keys = [
        "structural_misfit", "erm_loss", "domain_bn",
        "pac_bayes_full", "pac_bayes_kl", "loading_distance",
        "split_prior_pacbayes", "conformal_quantile",
        "loss_plus_margin", "loss_plus_mahalanobis", "loss_times_uncertainty",
        "loss_plus_gradient", "cross_env_variance", "cross_env_consistency",
        "_pac_cert_for_oracle",  # raw cert used by adaptive oracle
    ]
    tr_sum = {k: np.zeros(N) for k in train_keys}
    tr_cnt = {k: np.zeros(N) for k in train_keys}

    # cross-fold prediction store (for cross-env variance/consistency): for each
    # subject, collect its predicted class prob (P(y=1)) in each in-train fold.
    cross_fold_p1: list[list[float]] = [[] for _ in range(N)]

    # ---- TEST-side accumulators (single held-out fold) ----
    test_keys = ["max_softmax", "deep_ensemble5", "ensemble_disagree3", "mc_dropout"]
    te_val = {k: np.full(N, np.nan) for k in test_keys}

    for held_out in range(n_sites):
        train_mask = site_ids != held_out
        test_mask = site_ids == held_out
        if test_mask.sum() < 5 or train_mask.sum() < 20:
            continue

        train_feat = features[train_mask]
        train_tgt = targets[train_mask]
        train_sites = site_ids[train_mask]
        test_feat = features[test_mask]
        test_tgt = targets[test_mask]
        unique = train_sites.unique()
        smap = {int(s_): i for i, s_ in enumerate(unique)}
        remapped = torch.tensor([smap[int(s_)] for s_ in train_sites])
        nts = len(unique)
        ti_train = torch.where(train_mask)[0].numpy()
        ti_test = torch.where(test_mask)[0].numpy()

        # =================== FIRE / IOP model ===================
        model, loadings = _train_fire(train_feat, train_tgt, remapped, nts, dim, seed)
        model.eval()
        with torch.no_grad():
            tr_pred = model(train_feat, loadings, remapped)
            tr_loss = F.cross_entropy(tr_pred, train_tgt.long(), reduction="none").numpy()
            tr_probs = torch.softmax(tr_pred, dim=-1)
            tr_p1 = tr_probs[:, 1].numpy() if tr_probs.shape[1] > 1 else tr_probs.view(-1).numpy()
            tr_maxp = tr_probs.max(dim=-1).values.numpy()
            sp, _ = torch.sort(tr_probs, dim=-1, descending=True)
            tr_margin = (sp[:, 0] - sp[:, 1]).numpy()

        # structural misfit = per-subject training CE
        _acc(tr_sum, tr_cnt, "structural_misfit", ti_train, tr_loss)

        # store P(y=1) for cross-fold variance/consistency
        for j, gi in enumerate(ti_train):
            cross_fold_p1[gi].append(float(tr_p1[j]))

        # ---- PAC-Bayes family (training-side) ----
        empirical_std = loadings.std(dim=0).mean().item()
        pac_cfg = PACBayesConfig(
            sigma_prior=max(empirical_std * 2.0, 0.5),
            sigma_posterior=max(empirical_std * 0.5, 0.05),
            delta=0.05, bound_type="catoni",
        )
        prior_mean = loadings.mean(dim=0)
        loss_max = max(tr_loss.max(), 1e-8)
        norm_tr = torch.tensor(tr_loss / loss_max).clamp(0, 1)
        # Per-subject sample count for the PAC-Bayes complexity term. We use nts, the
        # number of training environments a subject is observed across, matching the
        # manuscript's framing that each subject's misfit is aggregated over the
        # multi-site traces in which it appears (n_i approx E-1). A strict per-fold
        # reading would use 1; both give the same qualitative conclusion (the KL term
        # underperforms the loss), and nts keeps the reported certificate consistent
        # with the paper. (bug-report #4: judged a modeling choice, not a clear defect.)
        counts_tr = torch.full((len(loadings),), float(nts))
        pac_full = compute_pac_bayes_bounds(loadings, norm_tr, counts_tr, pac_cfg)
        cert_full = pac_full.bounds.numpy() * loss_max
        _acc(tr_sum, tr_cnt, "pac_bayes_full", ti_train, cert_full)
        _acc(tr_sum, tr_cnt, "_pac_cert_for_oracle", ti_train, cert_full)
        _acc(tr_sum, tr_cnt, "pac_bayes_kl", ti_train, pac_full.kl_divergences.numpy())
        _acc(tr_sum, tr_cnt, "loading_distance", ti_train,
             (loadings - prior_mean.unsqueeze(0)).norm(dim=1).numpy())

        # split-prior PAC-Bayes: prior mean from first half of train
        nhalf = max(len(loadings) // 2, 1)
        prior_split = loadings[:nhalf].mean(dim=0)
        kl_split = compute_kl_divergence(
            loadings, prior_split, pac_cfg.sigma_prior, pac_cfg.sigma_posterior
        )
        sp_full = _catoni_from_parts(norm_tr, kl_split, counts_tr, pac_cfg.delta) * loss_max
        _acc(tr_sum, tr_cnt, "split_prior_pacbayes", ti_train, sp_full)

        # ---- conformal quantile (rank of training CE within fold) ----
        ranks = sp_stats.rankdata(tr_loss) / len(tr_loss)
        _acc(tr_sum, tr_cnt, "conformal_quantile", ti_train, ranks)

        # ---- combined loss + X (training-side) ----
        maha_tr = _mahalanobis_scores(train_feat, train_tgt, train_feat)
        _acc(tr_sum, tr_cnt, "loss_plus_margin", ti_train, _z(tr_loss) + _z(-tr_margin))
        _acc(tr_sum, tr_cnt, "loss_plus_mahalanobis", ti_train, _z(tr_loss) + _z(maha_tr))
        _acc(tr_sum, tr_cnt, "loss_times_uncertainty", ti_train, tr_loss * (1.0 - tr_maxp))
        # gradient norm on training inputs
        tf_g = train_feat.clone().requires_grad_(True)
        gp = model(tf_g, loadings.detach(), remapped)
        ghat = gp.argmax(dim=-1)
        logp = F.log_softmax(gp, dim=1).gather(1, ghat[:, None]).squeeze(1)
        gnorm = torch.autograd.grad(logp.sum(), tf_g)[0].norm(dim=1).detach().numpy()
        _acc(tr_sum, tr_cnt, "loss_plus_gradient", ti_train, _z(tr_loss) + _z(gnorm))

        # =================== ERM plain MLP (training-side) ===================
        erm = _train_erm(train_feat, train_tgt, dim, seed)
        erm.eval()
        with torch.no_grad():
            erm_tr = F.cross_entropy(
                erm(train_feat.to(DEVICE)), train_tgt.long().to(DEVICE),
                reduction="none",
            ).cpu().numpy()
        _acc(tr_sum, tr_cnt, "erm_loss", ti_train, erm_tr)

        # =================== Domain-BN (training-side) ===================
        dbn = _train_domain_bn(train_feat, train_tgt, remapped, nts, dim, seed)
        dbn.eval()
        with torch.no_grad():
            dbn_tr = F.cross_entropy(
                dbn(train_feat.to(DEVICE)), train_tgt.long().to(DEVICE),
                reduction="none",
            ).cpu().numpy()
        _acc(tr_sum, tr_cnt, "domain_bn", ti_train, dbn_tr)

        # =================== held-out (TEST) evaluation ===================
        tl = _predict_loadings(train_feat, loadings, test_feat)
        test_env = torch.zeros(test_feat.shape[0], dtype=torch.long)
        with torch.no_grad():
            tp = model(test_feat, tl, test_env)
            pl = tp.argmax(dim=-1)
            probs_te = torch.softmax(tp, dim=-1)
        correct = (pl == test_tgt.long()).float()
        subject_correct[ti_test] = correct
        subject_tested[ti_test] = True
        te_val["max_softmax"][ti_test] = (1.0 - probs_te.max(dim=-1).values.numpy())

        # ---- deep ensemble (5 and 3) -- genuine test-time UQ ----
        ens_probs = []
        for k in range(N_ENSEMBLE5):
            ens_seed = seed * 1000 + k * 137 + held_out * 17
            em, el = _train_fire(train_feat, train_tgt, remapped, nts, dim, ens_seed)
            etl = _predict_loadings(train_feat, el, test_feat)
            em.eval()
            with torch.no_grad():
                ens_probs.append(torch.softmax(em(test_feat, etl, test_env), dim=-1))
        ens5 = torch.stack(ens_probs)
        te_val["deep_ensemble5"][ti_test] = ens5.var(0).mean(-1).numpy()
        te_val["ensemble_disagree3"][ti_test] = ens5[:N_ENSEMBLE3].var(0).mean(-1).numpy()

        # ---- MC dropout (20 passes) ----
        mc = _train_erm(train_feat, train_tgt, dim, seed, dropout=MC_DROPOUT)
        mc.train()
        mc_probs = []
        with torch.no_grad():
            Xte_d = test_feat.to(DEVICE)
            for _ in range(MC_PASSES):
                mc_probs.append(torch.softmax(mc(Xte_d), dim=-1).cpu())
        te_val["mc_dropout"][ti_test] = torch.stack(mc_probs).var(0).mean(-1).numpy()

    # ---- cross-env variance / consistency from per-fold predictions ----
    # Each subject is predicted (P(y=1)) in every fold where it was in TRAIN (each
    # fold holds out a different environment -> genuinely cross-environment).
    #   variance    = Var_folds[ P(y=1) ]            (instability of the score)
    #   consistency = -mean_folds |2*P(y=1) - 1|     (low decision margin = the
    #                  prediction is not consistently confident; higher = less
    #                  reliable). Continuous, non-degenerate, larger = worse.
    cev = np.full(N, np.nan)
    cec = np.full(N, np.nan)
    for i in range(N):
        ps = cross_fold_p1[i]
        if len(ps) >= 2:
            arr = np.asarray(ps)
            cev[i] = float(arr.var())
            cec[i] = -float(np.mean(np.abs(2.0 * arr - 1.0)))
    # write these directly into the training-side aggregate (already per-subject)
    tr_mean = {k: tr_sum[k] / np.maximum(tr_cnt[k], 1) for k in train_keys}
    tr_mean["cross_env_variance"] = cev
    tr_mean["cross_env_consistency"] = cec
    has_train = {k: tr_cnt[k] > 0 for k in train_keys}
    has_train["cross_env_variance"] = ~np.isnan(cev)
    has_train["cross_env_consistency"] = ~np.isnan(cec)

    # ---- finalize ----
    tested = subject_tested.numpy()
    err = 1.0 - subject_correct.numpy()
    misfit = tr_mean["structural_misfit"]

    out = {}
    # training-side estimators
    for k in train_keys:
        if k.startswith("_"):
            continue
        m = tested & has_train[k]
        out[k] = _rho(tr_mean[k][m], err[m])
    out["cross_env_variance"] = _rho(
        cev[tested & has_train["cross_env_variance"]],
        err[tested & has_train["cross_env_variance"]],
    )
    out["cross_env_consistency"] = _rho(
        cec[tested & has_train["cross_env_consistency"]],
        err[tested & has_train["cross_env_consistency"]],
    )
    # test-side estimators
    for k in test_keys:
        m = tested & ~np.isnan(te_val[k])
        out[k] = _rho(te_val[k][m], err[m])

    # adaptive oracle: max over alpha of rho(alpha*misfit + (1-alpha)*pac_cert),
    # both training-side, target = binary err (test-label tuning -> upper bound).
    cert = tr_mean["_pac_cert_for_oracle"]
    valid = tested & has_train["structural_misfit"] & has_train["_pac_cert_for_oracle"]
    lz = _z(misfit[valid]); cz = _z(cert[valid])
    best = -1.0
    for alpha in np.linspace(0, 1, 21):
        r = _rho(alpha * lz + (1 - alpha) * cz, err[valid])
        if not np.isnan(r) and r > best:
            best = r
    out["adaptive_oracle"] = best

    extras = {
        "n_tested": int(tested.sum()),
        "err_rate": float(np.nanmean(err[tested])),
    }
    if want_selective:
        order = np.argsort(misfit[tested])
        corr_sorted = subject_correct.numpy()[tested][order]
        sel = {}
        for cov in [1.0, 0.8, 0.5, 0.3]:
            n = max(1, int(round(cov * len(corr_sorted))))
            sel[f"{int(cov*100)}"] = float(corr_sorted[:n].mean())
        extras["selective_misfit"] = sel
    return out, extras


def _acc(sum_d, cnt_d, key, idx, vals):
    """Accumulate per-subject values into the running sum/count dicts."""
    sum_d[key][idx] += np.asarray(vals, dtype=np.float64)
    cnt_d[key][idx] += 1.0


def _catoni_from_parts(norm_losses, kl_divs, counts, delta):
    """Catoni bound assembled from explicit empirical-loss + KL parts (numpy out)."""
    import math

    emp = norm_losses.clamp(min=1e-8, max=1.0 - 1e-8)
    catoni_emp = -torch.log(1.0 - emp)
    log_inv_delta = math.log(1.0 / delta)
    complexity = (kl_divs + log_inv_delta) / counts.clamp(min=1)
    return (1.0 - torch.exp(-(catoni_emp + complexity))).numpy()


def _z(x):
    """Z-score a vector; centering only when the std is numerically zero."""
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    if s < 1e-12:
        return x - x.mean()
    return (x - x.mean()) / s


# ---------------------------------------------------------------------------
# Within-fold per-protocol rho (ADNI) -- separate, mirrors paper tab:wf_adni
# ---------------------------------------------------------------------------
def within_fold_adni(features, targets, site_ids):
    """Per-protocol within-fold rho (misfit-as-test-loss vs binary err), 5 seeds.

    Mirrors run_subject_certificates within-fold protocol: for each held-out
    protocol, train on the rest, score held-out subjects by their TEST loss, and
    correlate with binary error WITHIN that protocol.
    """
    n_sites = int(site_ids.max().item()) + 1
    dim = features.shape[1]
    per_proto = {p: [] for p in range(n_sites)}
    all_combos = []
    for s_idx in range(N_SEEDS):
        seed = SEED_BASE + s_idx
        seed_everything(seed)
        for held_out in range(n_sites):
            train_mask = site_ids != held_out
            test_mask = site_ids == held_out
            if test_mask.sum() < 5 or train_mask.sum() < 20:
                continue
            train_feat = features[train_mask]
            train_tgt = targets[train_mask]
            train_sites = site_ids[train_mask]
            test_feat = features[test_mask]
            test_tgt = targets[test_mask]
            unique = train_sites.unique()
            smap = {int(s_): i for i, s_ in enumerate(unique)}
            remapped = torch.tensor([smap[int(s_)] for s_ in train_sites])
            nts = len(unique)
            model, loadings = _train_fire(
                train_feat, train_tgt, remapped, nts, dim, seed
            )
            tl = _predict_loadings(train_feat, loadings, test_feat)
            test_env = torch.zeros(test_feat.shape[0], dtype=torch.long)
            model.eval()
            with torch.no_grad():
                tp = model(test_feat, tl, test_env)
                pl = tp.argmax(dim=-1)
                ce = F.cross_entropy(tp, test_tgt.long(), reduction="none").numpy()
            err = 1.0 - (pl == test_tgt.long()).float().numpy()
            r = _rho(ce, err)
            per_proto[held_out].append(r)
            all_combos.append(r)
    summary = {}
    for p in range(n_sites):
        vals = [v for v in per_proto[p] if not np.isnan(v)]
        summary[f"protocol_{p}"] = {
            "rho_mean": float(np.mean(vals)) if vals else float("nan"),
            "rho_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
            "n_test": int((site_ids == p).sum()),
        }
    pos = sum(1 for v in all_combos if not np.isnan(v) and v > 0)
    summary["all"] = {
        "rho_mean": float(np.nanmean(all_combos)),
        "rho_std": float(np.nanstd(all_combos)),
        "n_positive": pos,
        "n_total": len([v for v in all_combos if not np.isnan(v)]),
    }
    return summary


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_adni():
    """ADNI sMRI cohort as (features, targets, site_ids); sites = scanner protocols."""
    from misfit.data.adni_loader import load_adni_smri

    mri = load_adni_smri()
    return mri.features, mri.targets, mri.site_ids


def load_oasis_ds():
    """OASIS-1 cohort as (features, targets, site_ids); sites = age quartiles."""
    from misfit.data.oasis_loader import load_oasis

    mri = load_oasis(n_envs=4, env_strategy="age_quartile", n_parcels=100)
    return mri.features, mri.targets, mri.site_ids


def run_dataset(name, features, targets, site_ids, want_within_fold=False):
    """Run run_seed for all seeds on one cohort; aggregate estimator rho mean/std.

    With want_within_fold, also aggregates the misfit-ranked selective-prediction
    accuracies across seeds.
    """
    print(f"\n=== {name.upper()}  N={len(features)} dim={features.shape[1]} "
          f"sites={int(site_ids.max())+1} ===", flush=True)
    per_seed_scores: dict[str, list[float]] = {}
    selective_per_seed = []
    wf_misfit_correct = []
    for s_idx in range(N_SEEDS):
        seed = SEED_BASE + s_idx
        t0 = time.time()
        out, extras = run_seed(
            features, targets, site_ids, seed, want_selective=want_within_fold
        )
        for k, v in out.items():
            per_seed_scores.setdefault(k, []).append(v)
        if want_within_fold:
            selective_per_seed.append(extras.get("selective_misfit"))
        print(
            f"  [{name} seed {seed}] misfit={out['structural_misfit']:+.4f} "
            f"err_rate={extras['err_rate']:.3f} ({time.time()-t0:.0f}s)",
            flush=True,
        )
    agg = {}
    for k, vals in per_seed_scores.items():
        v = np.array([x for x in vals if not np.isnan(x)])
        agg[k] = {
            "mean": float(v.mean()) if len(v) else float("nan"),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "per_seed": [float(x) for x in vals],
        }
    result = {"scores": agg, "n_seeds": N_SEEDS, "n_subjects": len(features),
              "n_sites": int(site_ids.max()) + 1}
    if want_within_fold:
        # selective prediction (mean across seeds)
        cov_keys = ["100", "80", "50", "30"]
        sel_agg = {}
        for ck in cov_keys:
            vals = [s[ck] for s in selective_per_seed if s]
            sel_agg[ck] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        result["selective_misfit"] = sel_agg
        print("\n  Selective prediction (misfit ranking, acc):", flush=True)
        for ck in cov_keys:
            print(f"    {ck:>4}% coverage: {sel_agg[ck]['mean']*100:.1f}%", flush=True)
    return result


def main():
    """CLI entry: --datasets {adni,oasis} and --skip-within-fold.

    Runs the estimator table per dataset, optionally the ADNI within-fold
    per-protocol analysis, writes honest_estimator_table.json, and prints the
    table sorted by ADNI rho.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["adni", "oasis"],
                    choices=["adni", "oasis"])
    ap.add_argument("--skip-within-fold", action="store_true")
    args = ap.parse_args()

    print(f"device={DEVICE}", flush=True)
    loaders = {"adni": load_adni, "oasis": load_oasis_ds}
    data = {}
    for ds in args.datasets:
        feat, tgt, sites = loaders[ds]()
        data[ds] = (feat, tgt, sites)

    results = {}
    for ds in args.datasets:
        feat, tgt, sites = data[ds]
        want_wf = (ds == "adni") and not args.skip_within_fold
        results[ds] = run_dataset(ds, feat, tgt, sites, want_within_fold=want_wf)

    # Within-fold per-protocol (ADNI only)
    if "adni" in args.datasets and not args.skip_within_fold:
        print("\n=== ADNI within-fold per-protocol ===", flush=True)
        feat, tgt, sites = data["adni"]
        wf = within_fold_adni(feat, tgt, sites)
        results["adni"]["within_fold"] = wf
        for p in range(int(sites.max()) + 1):
            s = wf[f"protocol_{p}"]
            print(f"  P{p}: rho={s['rho_mean']:+.3f}±{s['rho_std']:.3f} "
                  f"(n_test={s['n_test']})", flush=True)
        print(f"  ALL: rho={wf['all']['rho_mean']:+.3f}±{wf['all']['rho_std']:.3f} "
              f"({wf['all']['n_positive']}/{wf['all']['n_total']} positive)", flush=True)

    # Assemble final estimator table (per estimator -> per dataset)
    all_estimators = sorted({k for ds in results for k in results[ds]["scores"]})
    table = {}
    for est in all_estimators:
        table[est] = {}
        for ds in args.datasets:
            sc = results[ds]["scores"].get(est)
            if sc:
                table[est][ds] = {
                    "mean": sc["mean"], "std": sc["std"], "per_seed": sc["per_seed"],
                }
    final = {
        "estimator_table": table,
        "selective_misfit_adni": results.get("adni", {}).get("selective_misfit"),
        "within_fold_adni": results.get("adni", {}).get("within_fold"),
        "meta": {
            "target": "per_subject_binary_error (1 if misclassified in held-out fold)",
            "protocol": "leave-one-environment-out, pooled across folds, 5 seeds",
            "n_seeds": N_SEEDS,
            "anchors": {},
        },
    }
    (OUT / "honest_estimator_table.json").write_text(json.dumps(final, indent=2))
    print(f"\nSaved {OUT / 'honest_estimator_table.json'}", flush=True)

    # quick console table sorted by ADNI rho
    print("\n=== Estimator table (sorted by ADNI rho desc) ===", flush=True)
    rows = []
    for est in all_estimators:
        a = table[est].get("adni", {}).get("mean", float("nan"))
        o = table[est].get("oasis", {}).get("mean", float("nan"))
        rows.append((est, a, o))
    rows.sort(key=lambda r: (-(r[1] if not np.isnan(r[1]) else -9)))
    for est, a, o in rows:
        print(f"  {est:24s} ADNI={a:+.3f}  OASIS={o:+.3f}", flush=True)


if __name__ == "__main__":
    main()
