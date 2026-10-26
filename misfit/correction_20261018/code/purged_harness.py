"""Participant-purged variants of the historical ADNI LOSO loops.

run_seed_purged is a line-faithful copy of
misfit/experiments/run_honest_estimator_table.py::run_seed with
exactly these deltas (documented in report/EXECUTION_LOG.md):
  D1. PURGE: for held-out protocol e, every acquisition whose participant
      (PTID) appears in e's test set is removed from the training pool.
      Historical: train_mask = site_ids != held_out.
  D2. FOLD FEATURES: features are supplied by a callable so that V1 uses the
      historical global-PCA matrix unchanged and V2 re-fits PCA + z-norm on
      the purged training pool per fold (purged_loader.fold_pca).
  D3. EXTENDED RETURNS: per-scan arrays needed by the ml_reliability blocks
      (misfit, erm_loss, conformal_q, max_softmax, energy, mc_dropout,
      attribution, program-ablation), the per-scan count of contributing
      refits, and per-scan held-out error -- the historical run_seed returned
      only summary correlations.
All model/training/estimator code is IMPORTED UNCHANGED from the historical
module; nothing in the historical files is modified.

within_fold_purged mirrors within_fold_adni with the same purge.
collect_dynamics_purged mirrors run_training_dynamics.collect_dynamics.
"""

from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
sys.path.insert(0, _resolve_path("<REPO_ROOT>/misfit/experiments"))

from misfit.experiments import run_honest_estimator_table as H  # noqa: E402
from misfit.core.utils import seed_everything  # noqa: E402
from misfit.certificates.pac_bayes import (  # noqa: E402
    PACBayesConfig, compute_kl_divergence, compute_pac_bayes_bounds,
)
from run_training_dynamics import SCORES, train_log_dynamics  # noqa: E402

N_PROGRAMS = H.N_PROGRAMS


def purge_masks(site_ids: torch.Tensor, ptids: np.ndarray, held_out: int,
                seed: int | None = None):
    """D1: train excludes the held-out protocol AND all acquisitions of its
    test participants. (seed accepted for mask_fn interface symmetry; unused.)"""
    test_mask = site_ids == held_out
    pe = set(ptids[test_mask.numpy()])
    in_pe = torch.tensor(np.isin(ptids, sorted(pe)))
    train_mask = (site_ids != held_out) & ~in_pe
    return train_mask, test_mask


def random_removal_masks(site_ids: torch.Tensor, ptids: np.ndarray,
                         held_out: int, seed: int):
    """Addendum A8 control V1R: remove a uniformly random training subset of
    exactly the purge size for this fold (RandomState(1000 + seed), stream
    offset by held_out so folds differ); overlap participants may remain."""
    test_mask = site_ids == held_out
    hist_train = site_ids != held_out
    purged_train, _ = purge_masks(site_ids, ptids, held_out)
    n_remove = int(hist_train.sum() - purged_train.sum())
    rng = np.random.RandomState(1000 + seed + 97 * held_out)
    tr_idx = np.where(hist_train.numpy())[0]
    drop = rng.choice(tr_idx, size=n_remove, replace=False)
    train_mask = hist_train.clone()
    train_mask[torch.tensor(drop)] = False
    return train_mask, test_mask


def run_seed_purged(fold_features, targets, site_ids, ptids, seed,
                    want_selective=False, mask_fn=None):
    """One seed of the purged LOSO estimator loop (historical run_seed + D1-D3).

    Args:
        fold_features: callable (train_mask, test_mask) -> (train_feat,
            test_feat), keeping the V1/V2 feature construction outside the
            loop (D2).
        mask_fn: fold mask builder; defaults to purge_masks (D1).

    Returns:
        out: estimator name -> pooled Spearman rho against held-out error.
        extras: n_tested, err_rate, and the optional selective-accuracy table.
        perscan: per-scan arrays (D3) consumed by the summary scripts.
    """
    if mask_fn is None:
        mask_fn = purge_masks
    n_sites = int(site_ids.max().item()) + 1
    N = len(targets)
    dim = 50
    seed_everything(seed)

    subject_correct = torch.full((N,), float("nan"))
    subject_tested = torch.zeros(N, dtype=torch.bool)

    train_keys = [
        "structural_misfit", "erm_loss", "domain_bn",
        "pac_bayes_full", "pac_bayes_kl", "loading_distance",
        "split_prior_pacbayes", "conformal_quantile",
        "loss_plus_margin", "loss_plus_mahalanobis", "loss_times_uncertainty",
        "loss_plus_gradient", "cross_env_variance", "cross_env_consistency",
        "_pac_cert_for_oracle",
    ]
    tr_sum = {k: np.zeros(N) for k in train_keys}
    tr_cnt = {k: np.zeros(N) for k in train_keys}
    cross_fold_p1: list[list[float]] = [[] for _ in range(N)]
    test_keys = ["max_softmax", "deep_ensemble5", "ensemble_disagree3", "mc_dropout"]
    te_val = {k: np.full(N, np.nan) for k in test_keys}
    energy = np.full(N, np.nan)
    test_ce = np.full(N, np.nan)
    maha_test = np.full(N, np.nan)
    attr_sum = np.zeros((N, dim))
    prog_sum = np.zeros((N, N_PROGRAMS))

    for held_out in range(n_sites):
        train_mask, test_mask = mask_fn(site_ids, ptids, held_out, seed)
        if test_mask.sum() < 5 or train_mask.sum() < 20:
            continue
        train_feat, test_feat = fold_features(train_mask, test_mask)
        train_tgt = targets[train_mask]
        test_tgt = targets[test_mask]
        train_sites = site_ids[train_mask]
        unique = train_sites.unique()
        smap = {int(s_): i for i, s_ in enumerate(unique)}
        remapped = torch.tensor([smap[int(s_)] for s_ in train_sites])
        nts = len(unique)
        ti_train = torch.where(train_mask)[0].numpy()
        ti_test = torch.where(test_mask)[0].numpy()

        # =================== FIRE / IOP model ===================
        model, loadings = H._train_fire(train_feat, train_tgt, remapped, nts, dim, seed)
        model.eval()
        with torch.no_grad():
            tr_pred = model(train_feat, loadings, remapped)
            tr_loss = F.cross_entropy(tr_pred, train_tgt.long(), reduction="none").numpy()
            tr_probs = torch.softmax(tr_pred, dim=-1)
            tr_p1 = tr_probs[:, 1].numpy() if tr_probs.shape[1] > 1 else tr_probs.view(-1).numpy()
            tr_maxp = tr_probs.max(dim=-1).values.numpy()
            sp, _ = torch.sort(tr_probs, dim=-1, descending=True)
            tr_margin = (sp[:, 0] - sp[:, 1]).numpy()

        H._acc(tr_sum, tr_cnt, "structural_misfit", ti_train, tr_loss)
        for j, gi in enumerate(ti_train):
            cross_fold_p1[gi].append(float(tr_p1[j]))

        empirical_std = loadings.std(dim=0).mean().item()
        pac_cfg = PACBayesConfig(
            sigma_prior=max(empirical_std * 2.0, 0.5),
            sigma_posterior=max(empirical_std * 0.5, 0.05),
            delta=0.05, bound_type="catoni",
        )
        prior_mean = loadings.mean(dim=0)
        loss_max = max(tr_loss.max(), 1e-8)
        norm_tr = torch.tensor(tr_loss / loss_max).clamp(0, 1)
        counts_tr = torch.full((len(loadings),), float(nts))
        pac_full = compute_pac_bayes_bounds(loadings, norm_tr, counts_tr, pac_cfg)
        cert_full = pac_full.bounds.numpy() * loss_max
        H._acc(tr_sum, tr_cnt, "pac_bayes_full", ti_train, cert_full)
        H._acc(tr_sum, tr_cnt, "_pac_cert_for_oracle", ti_train, cert_full)
        H._acc(tr_sum, tr_cnt, "pac_bayes_kl", ti_train, pac_full.kl_divergences.numpy())
        H._acc(tr_sum, tr_cnt, "loading_distance", ti_train,
               (loadings - prior_mean.unsqueeze(0)).norm(dim=1).numpy())

        nhalf = max(len(loadings) // 2, 1)
        prior_split = loadings[:nhalf].mean(dim=0)
        kl_split = compute_kl_divergence(
            loadings, prior_split, pac_cfg.sigma_prior, pac_cfg.sigma_posterior
        )
        sp_full = H._catoni_from_parts(norm_tr, kl_split, counts_tr, pac_cfg.delta) * loss_max
        H._acc(tr_sum, tr_cnt, "split_prior_pacbayes", ti_train, sp_full)

        ranks = sp_stats.rankdata(tr_loss) / len(tr_loss)
        H._acc(tr_sum, tr_cnt, "conformal_quantile", ti_train, ranks)

        maha_tr = H._mahalanobis_scores(train_feat, train_tgt, train_feat)
        H._acc(tr_sum, tr_cnt, "loss_plus_margin", ti_train,
               H._z(tr_loss) + H._z(-tr_margin))
        H._acc(tr_sum, tr_cnt, "loss_plus_mahalanobis", ti_train,
               H._z(tr_loss) + H._z(maha_tr))
        H._acc(tr_sum, tr_cnt, "loss_times_uncertainty", ti_train, tr_loss * (1.0 - tr_maxp))
        tf_g = train_feat.clone().requires_grad_(True)
        gp = model(tf_g, loadings.detach(), remapped)
        ghat = gp.argmax(dim=-1)
        logp = F.log_softmax(gp, dim=1).gather(1, ghat[:, None]).squeeze(1)
        gnorm = torch.autograd.grad(logp.sum(), tf_g)[0].norm(dim=1).detach().numpy()
        H._acc(tr_sum, tr_cnt, "loss_plus_gradient", ti_train, H._z(tr_loss) + H._z(gnorm))

        # D3: input-grad attribution of the training CE (ml_reliability block C)
        tf_a = train_feat.clone().requires_grad_(True)
        ce_a = F.cross_entropy(model(tf_a, loadings.detach(), remapped),
                               train_tgt.long(), reduction="none")
        g_a = torch.autograd.grad(ce_a.sum(), tf_a)[0].detach().abs().numpy()
        attr_sum[ti_train] += g_a
        # D3: per-program ablation (ml_reliability block D)
        with torch.no_grad():
            for k in range(N_PROGRAMS):
                lk = loadings.clone(); lk[:, k] = 0.0
                ce_k = F.cross_entropy(model(train_feat, lk, remapped),
                                       train_tgt.long(), reduction="none").numpy()
                prog_sum[ti_train, k] += (ce_k - tr_loss)

        # =================== ERM plain MLP (training-side) ===================
        erm = H._train_erm(train_feat, train_tgt, dim, seed)
        erm.eval()
        with torch.no_grad():
            erm_tr = F.cross_entropy(
                erm(train_feat.to(H.DEVICE)), train_tgt.long().to(H.DEVICE),
                reduction="none",
            ).cpu().numpy()
        H._acc(tr_sum, tr_cnt, "erm_loss", ti_train, erm_tr)

        # =================== Domain-BN (training-side) ===================
        dbn = H._train_domain_bn(train_feat, train_tgt, remapped, nts, dim, seed)
        dbn.eval()
        with torch.no_grad():
            dbn_tr = F.cross_entropy(
                dbn(train_feat.to(H.DEVICE)), train_tgt.long().to(H.DEVICE),
                reduction="none",
            ).cpu().numpy()
        H._acc(tr_sum, tr_cnt, "domain_bn", ti_train, dbn_tr)

        # =================== held-out (TEST) evaluation ===================
        tl = H._predict_loadings(train_feat, loadings, test_feat)
        test_env = torch.zeros(test_feat.shape[0], dtype=torch.long)
        with torch.no_grad():
            tp = model(test_feat, tl, test_env)
            pl = tp.argmax(dim=-1)
            probs_te = torch.softmax(tp, dim=-1)
        correct = (pl == test_tgt.long()).float()
        subject_correct[ti_test] = correct
        subject_tested[ti_test] = True
        te_val["max_softmax"][ti_test] = (1.0 - probs_te.max(dim=-1).values.numpy())
        energy[ti_test] = (-torch.logsumexp(tp, dim=-1)).numpy()
        test_ce[ti_test] = F.cross_entropy(tp, test_tgt.long(),
                                           reduction="none").numpy()
        maha_test[ti_test] = H._mahalanobis_scores(train_feat, train_tgt, test_feat)

        ens_probs = []
        for k in range(H.N_ENSEMBLE5):
            ens_seed = seed * 1000 + k * 137 + held_out * 17
            em, el = H._train_fire(train_feat, train_tgt, remapped, nts, dim, ens_seed)
            etl = H._predict_loadings(train_feat, el, test_feat)
            em.eval()
            with torch.no_grad():
                ens_probs.append(torch.softmax(em(test_feat, etl, test_env), dim=-1))
        ens5 = torch.stack(ens_probs)
        te_val["deep_ensemble5"][ti_test] = ens5.var(0).mean(-1).numpy()
        te_val["ensemble_disagree3"][ti_test] = ens5[:H.N_ENSEMBLE3].var(0).mean(-1).numpy()

        mc = H._train_erm(train_feat, train_tgt, dim, seed, dropout=H.MC_DROPOUT)
        mc.train()
        mc_probs = []
        with torch.no_grad():
            Xte_d = test_feat.to(H.DEVICE)
            for _ in range(H.MC_PASSES):
                mc_probs.append(torch.softmax(mc(Xte_d), dim=-1).cpu())
        te_val["mc_dropout"][ti_test] = torch.stack(mc_probs).var(0).mean(-1).numpy()

    # cross-fold variance / consistency of the class-1 probability over the
    # folds in which each scan trained (needs >= 2 contributing folds)
    cev = np.full(N, np.nan)
    cec = np.full(N, np.nan)
    for i in range(N):
        ps = cross_fold_p1[i]
        if len(ps) >= 2:
            arr = np.asarray(ps)
            cev[i] = float(arr.var())
            cec[i] = -float(np.mean(np.abs(2.0 * arr - 1.0)))
    tr_mean = {k: tr_sum[k] / np.maximum(tr_cnt[k], 1) for k in train_keys}
    tr_mean["cross_env_variance"] = cev
    tr_mean["cross_env_consistency"] = cec
    has_train = {k: tr_cnt[k] > 0 for k in train_keys}
    has_train["cross_env_variance"] = ~np.isnan(cev)
    has_train["cross_env_consistency"] = ~np.isnan(cec)

    tested = subject_tested.numpy()
    err = 1.0 - subject_correct.numpy()
    misfit = tr_mean["structural_misfit"]

    out = {}
    for k in train_keys:
        if k.startswith("_"):
            continue
        m = tested & has_train[k]
        out[k] = H._rho(tr_mean[k][m], err[m])
    out["cross_env_variance"] = H._rho(
        cev[tested & has_train["cross_env_variance"]],
        err[tested & has_train["cross_env_variance"]])
    out["cross_env_consistency"] = H._rho(
        cec[tested & has_train["cross_env_consistency"]],
        err[tested & has_train["cross_env_consistency"]])
    for k in test_keys:
        m = tested & ~np.isnan(te_val[k])
        out[k] = H._rho(te_val[k][m], err[m])

    cert = tr_mean["_pac_cert_for_oracle"]
    valid = tested & has_train["structural_misfit"] & has_train["_pac_cert_for_oracle"]
    lz = H._z(misfit[valid]); cz = H._z(cert[valid])
    best = -1.0
    for alpha in np.linspace(0, 1, 21):
        r = H._rho(alpha * lz + (1 - alpha) * cz, err[valid])
        if not np.isnan(r) and r > best:
            best = r
    out["adaptive_oracle"] = best

    extras = {"n_tested": int(tested.sum()), "err_rate": float(np.nanmean(err[tested]))}
    if want_selective:
        order = np.argsort(misfit[tested])
        corr_sorted = subject_correct.numpy()[tested][order]
        sel = {}
        for cov in [1.0, 0.8, 0.5, 0.3]:
            n = max(1, int(round(cov * len(corr_sorted))))
            sel[f"{int(cov*100)}"] = float(corr_sorted[:n].mean())
        extras["selective_misfit"] = sel

    def _rank(x):
        r = np.full_like(x, np.nan, dtype=np.float64)
        m = ~np.isnan(x)
        r[m] = sp_stats.rankdata(x[m]) / m.sum()
        return r

    cnt = np.maximum(tr_cnt["structural_misfit"], 1.0)
    perscan = {
        "tested": tested, "err": err, "site": site_ids.numpy(),
        "misfit": misfit, "erm_loss": tr_mean["erm_loss"],
        "conformal_q": _rank(np.where(tr_cnt["structural_misfit"] > 0, misfit, np.nan)),
        "max_softmax": te_val["max_softmax"], "energy": energy,
        "mc_dropout": te_val["mc_dropout"],
        "test_ce": test_ce, "maha_test": maha_test,
        "deep_ensemble5": te_val["deep_ensemble5"],
        "attr": attr_sum / cnt[:, None], "prog": prog_sum / cnt[:, None],
        "n_contributing_refits": tr_cnt["structural_misfit"].astype(int),
    }
    return out, extras, perscan


def within_fold_purged(fold_features, targets, site_ids, ptids, mask_fn=None):
    """within_fold_adni with the purge; held-out test CE vs binary error."""
    if mask_fn is None:
        mask_fn = purge_masks
    n_sites = int(site_ids.max().item()) + 1
    dim = 50
    per_proto = {p: [] for p in range(n_sites)}
    all_combos = []
    for s_idx in range(H.N_SEEDS):
        seed = H.SEED_BASE + s_idx
        seed_everything(seed)
        for held_out in range(n_sites):
            train_mask, test_mask = mask_fn(site_ids, ptids, held_out, seed)
            if test_mask.sum() < 5 or train_mask.sum() < 20:
                continue
            train_feat, test_feat = fold_features(train_mask, test_mask)
            train_tgt = targets[train_mask]
            test_tgt = targets[test_mask]
            train_sites = site_ids[train_mask]
            unique = train_sites.unique()
            smap = {int(s_): i for i, s_ in enumerate(unique)}
            remapped = torch.tensor([smap[int(s_)] for s_ in train_sites])
            nts = len(unique)
            model, loadings = H._train_fire(train_feat, train_tgt, remapped, nts, dim, seed)
            tl = H._predict_loadings(train_feat, loadings, test_feat)
            test_env = torch.zeros(test_feat.shape[0], dtype=torch.long)
            model.eval()
            with torch.no_grad():
                tp = model(test_feat, tl, test_env)
                pl = tp.argmax(dim=-1)
                ce = F.cross_entropy(tp, test_tgt.long(), reduction="none").numpy()
            err = 1.0 - (pl == test_tgt.long()).float().numpy()
            r = H._rho(ce, err)
            per_proto[held_out].append(r)
            all_combos.append(r)
    summary = {}
    for p in range(n_sites):
        vals = [v for v in per_proto[p] if not np.isnan(v)]
        summary[f"protocol_{p}"] = {
            "rho_mean": float(np.mean(vals)) if vals else float("nan"),
            "rho_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals), "n_test": int((site_ids == p).sum()),
        }
    pos = sum(1 for v in all_combos if not np.isnan(v) and v > 0)
    summary["all"] = {
        "rho_mean": float(np.nanmean(all_combos)),
        "rho_std": float(np.nanstd(all_combos)),
        "n_positive": pos,
        "n_total": len([v for v in all_combos if not np.isnan(v)]),
    }
    return summary


def collect_dynamics_purged(fold_features, targets, site_ids, ptids, seed,
                            mask_fn=None):
    """run_training_dynamics.collect_dynamics with the purge."""
    if mask_fn is None:
        mask_fn = purge_masks
    n_sites = int(site_ids.max()) + 1
    N = len(targets)
    ssum = {k: np.zeros(N) for k in SCORES}
    scount = np.zeros(N)
    pred = np.full(N, -1)
    tested = np.zeros(N, bool)
    yall = targets.long().to(H.DEVICE)
    for held in range(n_sites):
        train_mask, test_mask = mask_fn(site_ids, ptids, held, seed)
        if test_mask.sum() < 5 or train_mask.sum() < 20:
            continue
        train_feat, test_feat = fold_features(train_mask, test_mask)
        idx_tr = np.where(train_mask.numpy())[0]
        te = test_mask.numpy()
        model, sc = train_log_dynamics(train_feat.to(H.DEVICE),
                                       yall[train_mask], seed)
        for k in SCORES:
            ssum[k][idx_tr] += sc[k]
        scount[idx_tr] += 1
        with torch.no_grad():
            pred[te] = model(test_feat.to(H.DEVICE)).argmax(1).cpu().numpy()
        tested[te] = True
    scores = {k: ssum[k] / np.maximum(scount, 1) for k in SCORES}
    err = (pred != targets.numpy().astype(int)).astype(float)
    return scores, err, tested, scount.astype(int)
