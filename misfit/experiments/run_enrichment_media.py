"""Enrichment experiments for the MedIA reliability paper.

Reuses the honest harness (run_honest_estimator_table.py): IOP recipe, binary
per-subject-error target, ADNI N=731 / 8 protocols, OASIS-1 N=226 / 4 quartiles,
5 seeds; scores are recomputed from the specified fits.

Three result blocks:
  1. SHIFT-SEVERITY: per held-out env shift magnitude (Mahalanobis of held-out
     mean under train covariance + 2-sample energy distance) vs per-env misfit
     rho, per-env best independent-UQ rho, per-env error rate. Does misfit's
     advantage over UQ GROW with shift magnitude?
  2. STRONGER BASELINES: energy score, temperature-scaled max-softmax,
     split-conformal nonconformity, deep-kNN in encoder rep, Mahalanobis in
     encoder rep. Does misfit still beat this stronger panel?
  3. CLINICAL UTILITY: reliability diagram + ECE for misfit-abstention,
     net-benefit / decision-curve, coverage-accuracy frontier.

Numerics (repo rules): float64 covariance; never torch.inverse (linalg.solve /
SVD pseudoinverse); condition-number guards (warn if kappa > 1e6).

Output: misfit/results/enrichment.{json,md}
"""

from __future__ import annotations
from misfit.paths import reporting_reference

from misfit.paths import resolve_path as _resolve_path

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))

# Reuse the honest harness machinery verbatim (same IOP recipe / construction).
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    DEVICE,
    MC_DROPOUT,
    MC_PASSES,
    N_ENSEMBLE3,
    N_ENSEMBLE5,
    N_SEEDS,
    SEED_BASE,
    _cond_warn,
    _mahalanobis_scores,
    _predict_loadings,
    _rho,
    _train_erm,
    _train_fire,
    load_adni,
    load_oasis_ds,
)
from misfit.core.utils import seed_everything  # noqa: E402

warnings.filterwarnings("ignore")

OUT = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
OUT.mkdir(parents=True, exist_ok=True)



# ===========================================================================
# Shift-magnitude proxies (float64, no torch.inverse)
# ===========================================================================
def _mahalanobis_mean_shift(train_X: np.ndarray, heldout_X: np.ndarray) -> float:
    """Mahalanobis distance of held-out env mean from train mean under train cov.

    d^2 = (mu_h - mu_t)^T Sigma_t^{-1} (mu_h - mu_t), solved via linalg.solve on a
    ridge-regularised float64 covariance (SVD pinv fallback). No torch.inverse.
    """
    Xt = np.asarray(train_X, dtype=np.float64)
    Xh = np.asarray(heldout_X, dtype=np.float64)
    d = Xt.shape[1]
    mu_t = Xt.mean(0)
    mu_h = Xh.mean(0)
    Xc = Xt - mu_t
    cov = (Xc.T @ Xc) / max(Xt.shape[0] - 1, 1) + 1e-3 * np.eye(d)
    _cond_warn(cov, "shift_mahalanobis_cov")
    diff = mu_h - mu_t
    try:
        sol = np.linalg.solve(cov, diff)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(cov, rcond=1e-12) @ diff
    d2 = float(diff @ sol)
    return float(np.sqrt(max(d2, 0.0)))


def _energy_distance(A: np.ndarray, B: np.ndarray, max_n: int = 2000) -> float:
    """Two-sample energy distance E = 2 E||a-b|| - E||a-a'|| - E||b-b'||.

    Subsamples to max_n per side for tractable O(n^2) pairwise norms. Deterministic
    given a fixed RNG seed for subsampling. Cross-check for the Mahalanobis proxy.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    rng = np.random.default_rng(0)
    if A.shape[0] > max_n:
        A = A[rng.choice(A.shape[0], max_n, replace=False)]
    if B.shape[0] > max_n:
        B = B[rng.choice(B.shape[0], max_n, replace=False)]

    def _mean_pairwise(P, Q):
        # mean Euclidean distance between rows of P and rows of Q
        d2 = (
            (P * P).sum(1)[:, None]
            + (Q * Q).sum(1)[None, :]
            - 2.0 * (P @ Q.T)
        )
        np.maximum(d2, 0.0, out=d2)
        return float(np.sqrt(d2).mean())

    d_ab = _mean_pairwise(A, B)
    d_aa = _mean_pairwise(A, A)
    d_bb = _mean_pairwise(B, B)
    e = 2.0 * d_ab - d_aa - d_bb
    return float(max(e, 0.0))


# ===========================================================================
# EXPERIMENT 1: shift-severity characterisation (per held-out environment)
# ===========================================================================
def shift_severity(name, features, targets, site_ids):
    """Per-env misfit rho, best-UQ rho, error rate, and two shift proxies.

    For each held-out env: train IOP on the rest (same recipe as the harness),
    score held-out subjects by their TEST cross-entropy (= deployable misfit at
    test time) AND by genuine test-time UQ (max-softmax, deep-ensemble
    disagreement). Per-env rho is computed within each env per seed, then averaged
    over 5 seeds. Shift proxies are feature-only (deterministic, seed-free).
    """
    n_sites = int(site_ids.max().item()) + 1
    dim = features.shape[1]

    # per-env, per-seed rho accumulators
    misfit_rho = {e: [] for e in range(n_sites)}
    maxsoft_rho = {e: [] for e in range(n_sites)}
    ens_rho = {e: [] for e in range(n_sites)}
    err_rate = {e: [] for e in range(n_sites)}

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
            model.eval()
            tl = _predict_loadings(train_feat, loadings, test_feat)
            test_env = torch.zeros(test_feat.shape[0], dtype=torch.long)
            with torch.no_grad():
                tp = model(test_feat, tl, test_env)
                pl = tp.argmax(dim=-1)
                probs_te = torch.softmax(tp, dim=-1)
                ce = F.cross_entropy(tp, test_tgt.long(), reduction="none").numpy()
            err = 1.0 - (pl == test_tgt.long()).float().numpy()
            maxsoft = 1.0 - probs_te.max(dim=-1).values.numpy()

            # deep-ensemble disagreement (3) -- genuine test-time UQ
            ens_probs = []
            for k in range(N_ENSEMBLE5):
                ens_seed = seed * 1000 + k * 137 + held_out * 17
                em, el = _train_fire(
                    train_feat, train_tgt, remapped, nts, dim, ens_seed
                )
                etl = _predict_loadings(train_feat, el, test_feat)
                em.eval()
                with torch.no_grad():
                    ens_probs.append(
                        torch.softmax(em(test_feat, etl, test_env), dim=-1)
                    )
            ens5 = torch.stack(ens_probs)
            ens_disagree = ens5[:N_ENSEMBLE3].var(0).mean(-1).numpy()

            misfit_rho[held_out].append(_rho(ce, err))
            maxsoft_rho[held_out].append(_rho(maxsoft, err))
            ens_rho[held_out].append(_rho(ens_disagree, err))
            err_rate[held_out].append(float(err.mean()))

    # ---- per-env summary ----
    feats_np = features.numpy().astype(np.float64)
    rows = []
    for e in range(n_sites):
        m = (site_ids == e).numpy()
        if m.sum() < 5:
            continue
        train_np = feats_np[~m]
        held_np = feats_np[m]
        maha = _mahalanobis_mean_shift(train_np, held_np)
        energy = _energy_distance(train_np, held_np)

        def _ms(d):
            v = np.array([x for x in d[e] if not np.isnan(x)])
            return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)

        mf_m, mf_s = _ms(misfit_rho)
        ux_m, ux_s = _ms(maxsoft_rho)
        ue_m, ue_s = _ms(ens_rho)
        er_m = float(np.mean(err_rate[e]))
        best_uq = max(ux_m, ue_m)
        rows.append(
            {
                "env": e,
                "n_test": int(m.sum()),
                "maha_shift": maha,
                "energy_shift": energy,
                "misfit_rho": mf_m,
                "misfit_rho_std": mf_s,
                "maxsoft_rho": ux_m,
                "ensemble_rho": ue_m,
                "best_uq_rho": best_uq,
                "gap_misfit_minus_uq": mf_m - best_uq,
                "err_rate": er_m,
            }
        )
    return rows


def _spear(x, y):
    """Spearman (rho, p, n) over finite pairs; nan when n < 3 or a side is constant."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return float("nan"), float("nan"), int(ok.sum())
    r, p = sp_stats.spearmanr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def analyze_shift(rows_by_ds):
    """Correlate shift magnitude with misfit rho, UQ rho, and the gap.

    Per dataset (n = #envs) and pooled across datasets (the 8+4 = 12 envs the
    prompt anticipates). Small n -> a trend, not inferential.
    """
    out = {}
    pooled = {k: [] for k in
              ["maha", "energy", "misfit", "uq", "gap", "err"]}
    for ds, rows in rows_by_ds.items():
        maha = [r["maha_shift"] for r in rows]
        energy = [r["energy_shift"] for r in rows]
        mf = [r["misfit_rho"] for r in rows]
        uq = [r["best_uq_rho"] for r in rows]
        gap = [r["gap_misfit_minus_uq"] for r in rows]
        err = [r["err_rate"] for r in rows]
        pooled["maha"] += maha
        pooled["energy"] += energy
        pooled["misfit"] += mf
        pooled["uq"] += uq
        pooled["gap"] += gap
        pooled["err"] += err

        def _c(a, b):
            r, p, n = _spear(a, b)
            return {"rho": r, "p": p, "n": n}

        out[ds] = {
            "n_envs": len(rows),
            "maha_vs_misfit": _c(maha, mf),
            "maha_vs_uq": _c(maha, uq),
            "maha_vs_gap": _c(maha, gap),
            "maha_vs_err": _c(maha, err),
            "energy_vs_misfit": _c(energy, mf),
            "energy_vs_uq": _c(energy, uq),
            "energy_vs_gap": _c(energy, gap),
            "maha_vs_energy_consistency": _c(maha, energy),
        }

    def _c(a, b):
        r, p, n = _spear(a, b)
        return {"rho": r, "p": p, "n": n}

    out["pooled"] = {
        "n_envs": len(pooled["maha"]),
        "maha_vs_misfit": _c(pooled["maha"], pooled["misfit"]),
        "maha_vs_uq": _c(pooled["maha"], pooled["uq"]),
        "maha_vs_gap": _c(pooled["maha"], pooled["gap"]),
        "maha_vs_err": _c(pooled["maha"], pooled["err"]),
        "energy_vs_gap": _c(pooled["energy"], pooled["gap"]),
        "maha_vs_energy_consistency": _c(pooled["maha"], pooled["energy"]),
    }
    return out


# ===========================================================================
# EXPERIMENT 2: stronger baseline panel (test-time, per-subject, 5 seeds)
# ===========================================================================
def _temperature_scale(logits_cal, y_cal):
    """Fit a single temperature T>0 by minimising NLL on a held-in cal split.

    Standard Guo et al. 2017 temperature scaling. Returns T (float).
    """
    logits = torch.as_tensor(logits_cal, dtype=torch.float64)
    y = torch.as_tensor(y_cal, dtype=torch.long)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)

    def _closure():
        opt.zero_grad()
        T = torch.exp(logT)
        loss = F.cross_entropy(logits / T, y)
        loss.backward()
        return loss

    opt.step(_closure)
    return float(torch.exp(logT).item())


def _deep_knn_scores(train_rep, test_rep, k=10):
    """Mean distance to k nearest training reps (deep-kNN OOD score, float64)."""
    A = np.asarray(test_rep, dtype=np.float64)
    B = np.asarray(train_rep, dtype=np.float64)
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    dist = np.sqrt(d2)
    kk = min(k, dist.shape[1])
    part = np.partition(dist, kk - 1, axis=1)[:, :kk]
    return part.mean(1)


def _mahalanobis_rep(train_rep, train_y, test_rep):
    """Mahalanobis in the ENCODER representation (class-conditional, float64)."""
    return _mahalanobis_scores(
        torch.as_tensor(train_rep, dtype=torch.float32),
        torch.as_tensor(np.asarray(train_y), dtype=torch.float32),
        torch.as_tensor(test_rep, dtype=torch.float32),
    )


def stronger_panel_seed(features, targets, site_ids, seed):
    """One seed: LOO folds, test-time scores for the stronger baseline panel.

    Scores (per held-out subject, correlated with binary error pooled over folds):
      structural_misfit_test : test CE of the IOP model (deployable misfit)
      energy_score           : -logsumexp(test logits)
      temp_max_softmax       : 1 - max softmax after temperature scaling (T from
                               a held-in cal split of train)
      split_conformal        : split-conformal nonconformity = test CE ranked
                               against a proper calibration split's CE quantiles
      deep_knn_rep           : mean dist to k-NN train reps in encoder space
      mahalanobis_rep        : class-cond Mahalanobis in encoder space
    """
    n_sites = int(site_ids.max().item()) + 1
    dim = features.shape[1]
    N = features.shape[0]
    seed_everything(seed)

    keys = [
        "structural_misfit_test", "energy_score", "temp_max_softmax",
        "split_conformal", "deep_knn_rep", "mahalanobis_rep",
    ]
    te = {k: np.full(N, np.nan) for k in keys}
    correct = np.full(N, np.nan)
    tested = np.zeros(N, dtype=bool)

    # deployable training-side misfit (mean CE over in-train folds) -- the
    # canonical anchor the paper reports; recomputed here for an apples-to-apples
    # comparison against the label-free test-time baseline panel.
    tr_sum = np.zeros(N)
    tr_cnt = np.zeros(N)

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

        model, loadings = _train_fire(train_feat, train_tgt, remapped, nts, dim, seed)
        model.eval()
        with torch.no_grad():
            tr_pred = model(train_feat, loadings, remapped)
            tr_loss = F.cross_entropy(
                tr_pred, train_tgt.long(), reduction="none"
            ).numpy()
        tr_sum[ti_train] += tr_loss
        tr_cnt[ti_train] += 1.0
        tl = _predict_loadings(train_feat, loadings, test_feat)
        test_env = torch.zeros(test_feat.shape[0], dtype=torch.long)
        with torch.no_grad():
            tp = model(test_feat, tl, test_env)
            pl = tp.argmax(dim=-1)
            ce = F.cross_entropy(tp, test_tgt.long(), reduction="none").numpy()
            logits_te = tp.numpy().astype(np.float64)
            # encoder representations h_theta(x)
            rep_tr = model.encoder(train_feat).numpy().astype(np.float64)
            rep_te = model.encoder(test_feat).numpy().astype(np.float64)

        correct[ti_test] = (pl == test_tgt.long()).float().numpy()
        tested[ti_test] = True

        # --- deployable misfit at test time (test CE) ---
        te["structural_misfit_test"][ti_test] = ce

        # --- energy score = -logsumexp(logits) (Liu et al. 2020) ---
        te["energy_score"][ti_test] = -_logsumexp_np(logits_te)

        # --- temperature-scaled max-softmax ---
        # cal split: hold out a class-stratified 30% of train for T fitting
        rng = np.random.default_rng(seed * 7 + held_out)
        n_tr = train_feat.shape[0]
        cal_idx = rng.choice(n_tr, max(int(0.3 * n_tr), 10), replace=False)
        with torch.no_grad():
            tl_cal = _predict_loadings(
                train_feat, loadings, train_feat[cal_idx]
            )
            env_cal = torch.zeros(len(cal_idx), dtype=torch.long)
            logits_cal = model(
                train_feat[cal_idx], tl_cal, env_cal
            ).numpy().astype(np.float64)
        T = _temperature_scale(logits_cal, train_tgt[cal_idx].long().numpy())
        scaled = torch.softmax(
            torch.as_tensor(logits_te / max(T, 1e-3)), dim=-1
        ).numpy()
        te["temp_max_softmax"][ti_test] = 1.0 - scaled.max(1)

        # --- split-conformal nonconformity (proper calibration split) ---
        # calibration CE scores on the held-in cal split; nonconformity of a
        # test point = fraction of cal scores <= its score (rank-based p-value),
        # the standard split-conformal score. Larger = more nonconforming.
        with torch.no_grad():
            ce_cal = F.cross_entropy(
                model(train_feat[cal_idx], tl_cal, env_cal),
                train_tgt[cal_idx].long(),
                reduction="none",
            ).numpy()
        ce_cal_sorted = np.sort(ce_cal)
        ranks = np.searchsorted(ce_cal_sorted, ce, side="right") / (
            len(ce_cal_sorted) + 1.0
        )
        te["split_conformal"][ti_test] = ranks

        # --- deep-kNN distance in encoder representation ---
        te["deep_knn_rep"][ti_test] = _deep_knn_scores(rep_tr, rep_te, k=10)

        # --- Mahalanobis in encoder representation ---
        te["mahalanobis_rep"][ti_test] = _mahalanobis_rep(
            rep_tr, train_tgt.numpy(), rep_te
        )

    err = 1.0 - correct
    out = {}
    for k in keys:
        m = tested & np.isfinite(te[k])
        out[k] = _rho(te[k][m], err[m])
    # deployable training-side misfit (the canonical anchor)
    misfit_tr = tr_sum / np.maximum(tr_cnt, 1)
    m = tested & (tr_cnt > 0)
    out["structural_misfit_train"] = _rho(misfit_tr[m], err[m])
    return out


def _logsumexp_np(x):
    """Row-wise logsumexp with the max-subtraction trick (numerical stability)."""
    m = x.max(axis=1, keepdims=True)
    return (m.squeeze(1) + np.log(np.exp(x - m).sum(axis=1)))


def stronger_panel(name, features, targets, site_ids):
    """Run stronger_panel_seed over all seeds; aggregate mean/std per estimator."""
    per_seed = {}
    for s_idx in range(N_SEEDS):
        seed = SEED_BASE + s_idx
        t0 = time.time()
        out = stronger_panel_seed(features, targets, site_ids, seed)
        for k, v in out.items():
            per_seed.setdefault(k, []).append(v)
        print(
            f"  [{name} panel seed {seed}] "
            f"misfit_train={out['structural_misfit_train']:+.3f} "
            f"misfit_test={out['structural_misfit_test']:+.3f} "
            f"energy={out['energy_score']:+.3f} "
            f"conformal={out['split_conformal']:+.3f} ({time.time()-t0:.0f}s)",
            flush=True,
        )
    agg = {}
    for k, vals in per_seed.items():
        v = np.array([x for x in vals if not np.isnan(x)])
        agg[k] = {
            "mean": float(v.mean()) if len(v) else float("nan"),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "per_seed": [float(x) for x in vals],
        }
    return agg


# ===========================================================================
# EXPERIMENT 3: clinical utility of misfit-based selective prediction
# ===========================================================================
def clinical_utility(name, features, targets, site_ids, n_bins=10):
    """Reliability/ECE + net-benefit + coverage-accuracy for misfit abstention.

    Pools per-subject (training-derived) structural misfit and binary error over
    LOO folds, 5 seeds (same as the harness misfit / selective_misfit). The misfit
    score is the deployable training-side per-subject CE (mean over in-train
    folds), the exact signal the paper's selective-prediction table uses.
    """
    n_sites = int(site_ids.max().item()) + 1
    dim = features.shape[1]
    N = features.shape[0]

    # accumulate per-subject training-side misfit (mean over in-train folds) and
    # the single held-out binary error, per seed; pool seeds for calibration.
    all_misfit = []
    all_err = []
    sel_per_seed = []

    for s_idx in range(N_SEEDS):
        seed = SEED_BASE + s_idx
        seed_everything(seed)
        tr_sum = np.zeros(N)
        tr_cnt = np.zeros(N)
        subject_correct = np.full(N, np.nan)
        tested = np.zeros(N, dtype=bool)
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

            model, loadings = _train_fire(
                train_feat, train_tgt, remapped, nts, dim, seed
            )
            model.eval()
            with torch.no_grad():
                tr_pred = model(train_feat, loadings, remapped)
                tr_loss = F.cross_entropy(
                    tr_pred, train_tgt.long(), reduction="none"
                ).numpy()
            tr_sum[ti_train] += tr_loss
            tr_cnt[ti_train] += 1.0
            tl = _predict_loadings(train_feat, loadings, test_feat)
            test_env = torch.zeros(test_feat.shape[0], dtype=torch.long)
            with torch.no_grad():
                tp = model(test_feat, tl, test_env)
                pl = tp.argmax(dim=-1)
            subject_correct[ti_test] = (pl == test_tgt.long()).float().numpy()
            tested[ti_test] = True

        misfit = tr_sum / np.maximum(tr_cnt, 1)
        m = tested & (tr_cnt > 0)
        mf = misfit[m]
        err = 1.0 - subject_correct[m]
        all_misfit.append(mf)
        all_err.append(err)

        # selective prediction (accept lowest-misfit fraction)
        order = np.argsort(mf)
        corr_sorted = (1.0 - err)[order]
        sel = {}
        for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
            n = max(1, int(round(cov * len(corr_sorted))))
            sel[f"{int(cov*100)}"] = float(corr_sorted[:n].mean())
        sel_per_seed.append(sel)

    # ---- pool seeds for reliability / ECE / net-benefit ----
    mf_all = np.concatenate(all_misfit)
    err_all = np.concatenate(all_err)

    # reliability diagram: bin by misfit quantile, empirical error per bin
    # (misfit as an abstention/risk score -> should track empirical error rate)
    order = np.argsort(mf_all)
    mf_sorted = mf_all[order]
    err_sorted = err_all[order]
    n = len(mf_sorted)
    edges = [int(round(i * n / n_bins)) for i in range(n_bins + 1)]
    bins = []
    ece = 0.0
    # ECE here treats the (min-max normalised) misfit as a predicted error prob.
    mf_norm = (mf_sorted - mf_sorted.min()) / max(
        mf_sorted.max() - mf_sorted.min(), 1e-12
    )
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        pred = float(mf_norm[lo:hi].mean())
        emp = float(err_sorted[lo:hi].mean())
        w = (hi - lo) / n
        ece += w * abs(pred - emp)
        bins.append(
            {
                "bin": b,
                "misfit_mean": float(mf_sorted[lo:hi].mean()),
                "misfit_norm_mean": pred,
                "empirical_error": emp,
                "n": int(hi - lo),
            }
        )
    # monotonicity of empirical error across misfit bins (Spearman bin-index vs err)
    bidx = [bb["bin"] for bb in bins]
    berr = [bb["empirical_error"] for bb in bins]
    mono_rho, mono_p, _ = _spear(bidx, berr)

    # ---- decision-curve / net-benefit ----
    # Frame: ABSTENTION decision. "Treat" analogue = flag-as-unreliable (abstain).
    # A subject should be abstained iff it would be an error. Net benefit of the
    # misfit rule at threshold t (abstain if misfit-percentile >= t):
    #   NB(t) = TP/N - FP/N * (p_t / (1 - p_t))
    # where TP = abstained-and-error, FP = abstained-and-correct, p_t the abstain
    # threshold odds. Compared against treat-none (NB=0) and treat-all baselines.
    pct = sp_stats.rankdata(mf_all) / len(mf_all)  # misfit percentile
    is_err = err_all > 0.5
    prevalence = float(is_err.mean())
    nb_curve = []
    for pt in np.linspace(0.05, 0.6, 12):
        abstain = pct >= (1.0 - pt)  # flag top-pt fraction by misfit
        tp = float((abstain & is_err).sum())
        fp = float((abstain & ~is_err).sum())
        nb = tp / len(err_all) - fp / len(err_all) * (pt / (1.0 - pt))
        # treat-all (abstain everyone) net benefit at same threshold
        nb_all = prevalence - (1.0 - prevalence) * (pt / (1.0 - pt))
        nb_curve.append(
            {
                "p_t": float(pt),
                "net_benefit_misfit": float(nb),
                "net_benefit_treat_all": float(nb_all),
                "net_benefit_treat_none": 0.0,
                "advantage_over_best_default": float(nb - max(nb_all, 0.0)),
            }
        )
    positive_nb = all(
        c["net_benefit_misfit"] >= max(c["net_benefit_treat_all"], 0.0) - 1e-9
        for c in nb_curve
    )

    # ---- coverage-accuracy frontier (mean across seeds) ----
    cov_keys = ["100", "90", "80", "70", "60", "50", "40", "30"]
    frontier = {}
    for ck in cov_keys:
        vals = [s[ck] for s in sel_per_seed]
        frontier[ck] = {
            "acc_mean": float(np.mean(vals)),
            "acc_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }

    return {
        "ece": float(ece),
        "reliability_bins": bins,
        "monotonicity_rho": mono_rho,
        "monotonicity_p": mono_p,
        "prevalence_error": prevalence,
        "net_benefit_curve": nb_curve,
        "net_benefit_positive_all_thresholds": bool(positive_nb),
        "coverage_accuracy": frontier,
        "n_pooled": int(len(mf_all)),
    }


# ===========================================================================
# Verdicts
# ===========================================================================
def verdict_shift(analysis):
    """Does misfit's advantage over UQ grow with shift magnitude?"""
    p = analysis["pooled"]["maha_vs_gap"]["rho"]
    pe = analysis["pooled"]["maha_vs_energy_consistency"]["rho"]
    pm = analysis["pooled"]["maha_vs_misfit"]["rho"]
    pu = analysis["pooled"]["maha_vs_uq"]["rho"]
    parts = [
        f"pooled(12 env) Mahalanobis-vs-(misfit-UQ gap) rho={p:+.2f}",
        f"misfit rho={pm:+.2f}, UQ rho={pu:+.2f}",
        f"Maha-Energy proxy agreement rho={pe:+.2f}",
    ]
    if np.isfinite(p) and p >= 0.4:
        v = "STRONG: misfit-UQ gap grows with shift -- add as a selling point."
    elif np.isfinite(p) and p >= 0.2:
        v = "MODEST trend: gap grows with shift but n is small; report as trend."
    else:
        v = "NOT robust: gap does not clearly grow with shift -- skip as a claim."
    return "; ".join(parts) + ". VERDICT: " + v


def verdict_panel(panel_by_ds):
    """Compare the label-free baseline panel against the DEPLOYABLE training-side
    misfit (the anchor the paper reports). test-CE misfit is shown as a reference
    upper bound only and is not the thing baselines must beat."""
    lines = []
    add = True
    assessed = True
    misfit_keys = {"structural_misfit_train", "structural_misfit_test"}
    for ds, agg in panel_by_ds.items():
        # Prefer the current run; use a supplied historical reference only if needed.
        anchor = agg.get("structural_misfit_train", {}).get("mean")
        if anchor is None:
            anchor = (reporting_reference('misfit_anchor', required=False) or {}).get(ds)
        if anchor is None or not np.isfinite(anchor):
            lines.append(f'{ds.upper()}: baseline comparison not assessed; no reference supplied')
            assessed = False
            continue
        beaters = []
        for k, v in agg.items():
            if k in misfit_keys:
                continue
            if np.isfinite(v["mean"]) and v["mean"] > anchor + 0.01:
                beaters.append(f"{k}={v['mean']:+.3f}")
        mtr = agg.get("structural_misfit_train", {}).get("mean", float("nan"))
        seg = (
            f"{ds.upper()}: deployable misfit(train)={mtr:+.3f}; "
            + ("beaten by " + ", ".join(beaters) if beaters
               else "beats every label-free baseline in the stronger panel")
        )
        lines.append(seg)
        if beaters:
            add = False
    if not assessed:
        return '; '.join(lines) + '. VERDICT: comparison not assessed for all datasets.'
    v = (
        "STRONG: deployable misfit still beats the stronger label-free panel on "
        "both datasets -- keep as a selling point."
        if add
        else "MIXED: at least one stronger baseline matches/beats deployable "
        "misfit -- report honestly, do not overclaim."
    )
    return "; ".join(lines) + ". VERDICT: " + v


def verdict_clinical(clin_by_ds):
    """One-line verdict on calibration, net benefit and the coverage-accuracy gain."""
    lines = []
    ok = True
    for ds, c in clin_by_ds.items():
        f = c["coverage_accuracy"]
        a100 = f["100"]["acc_mean"] * 100
        a50 = f["50"]["acc_mean"] * 100
        lines.append(
            f"{ds.upper()}: ECE={c['ece']:.3f}, "
            f"reliability monotonic rho={c['monotonicity_rho']:+.2f}, "
            f"net-benefit>=default at all t={c['net_benefit_positive_all_thresholds']}, "
            f"cov-acc {a100:.1f}->{a50:.1f}@50%"
        )
        if not (
            c["ece"] < 0.15
            and c["net_benefit_positive_all_thresholds"]
            and (a50 - a100) > 2.0
        ):
            ok = False
    v = (
        "STRONG: misfit-abstention is well-calibrated with positive clinical net "
        "benefit -- add as a selling point."
        if ok
        else "PARTIAL: coverage-accuracy gain is real but calibration/net-benefit "
        "is imperfect -- report with caveats."
    )
    return "; ".join(lines) + ". VERDICT: " + v


# ===========================================================================
# Markdown report
# ===========================================================================
def write_md(result):
    """Render the three result blocks and verdicts into enrichment.md."""
    L = []
    L.append("# MedIA reliability paper -- enrichment experiments\n")
    L.append(
        "Reuses the honest IOP harness (binary per-subject-error target, "
        "leave-one-environment-out, 5 seeds). Comparisons use the current "
        "run and any explicitly supplied historical references.\n"
    )
    L.append(
        "Numerics: float64 covariance, no torch.inverse (linalg.solve / SVD pinv), "
        "condition-number guards.\n"
    )

    # ---- Block 1 ----
    L.append("\n## Block 1 -- Shift-severity characterisation\n")
    L.append(
        "Per held-out environment: two feature-space shift proxies (Mahalanobis "
        "of the held-out mean under the train covariance; 2-sample energy "
        "distance), per-env test-time misfit rho, per-env best independent-UQ rho "
        "(max-softmax / deep-ensemble disagreement), per-env error rate.\n"
    )
    for ds, rows in result["shift_severity"]["per_env"].items():
        L.append(f"\n### {ds.upper()} ({len(rows)} environments)\n")
        L.append(
            "| env | n | Mahal shift | energy shift | misfit rho | best UQ rho | "
            "misfit-UQ gap | err rate |"
        )
        L.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda z: z["maha_shift"]):
            L.append(
                f"| {r['env']} | {r['n_test']} | {r['maha_shift']:.2f} | "
                f"{r['energy_shift']:.3f} | {r['misfit_rho']:+.3f} | "
                f"{r['best_uq_rho']:+.3f} | {r['gap_misfit_minus_uq']:+.3f} | "
                f"{r['err_rate']:.3f} |"
            )
    a = result["shift_severity"]["analysis"]
    L.append("\n**Spearman correlations across environments (small n -> trend):**\n")
    L.append("| set | n_env | Mahal-vs-misfit | Mahal-vs-UQ | Mahal-vs-gap | "
             "Mahal-vs-err | Mahal-vs-energy |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ["adni", "oasis", "pooled"]:
        if k not in a:
            continue
        s = a[k]
        def g(key):
            return s.get(key, {}).get("rho", float("nan"))
        L.append(
            f"| {k} | {s['n_envs']} | {g('maha_vs_misfit'):+.2f} | "
            f"{g('maha_vs_uq'):+.2f} | {g('maha_vs_gap'):+.2f} | "
            f"{g('maha_vs_err'):+.2f} | {g('maha_vs_energy_consistency'):+.2f} |"
        )
    L.append(f"\n**Verdict (Block 1):** {result['shift_severity']['verdict']}\n")

    # ---- Block 2 ----
    L.append("\n## Block 2 -- Stronger baseline panel\n")
    L.append(
        "Same binary-error folds. Per-subject rho (5 seeds) for: energy score "
        "(-logsumexp logits), temperature-scaled max-softmax (T fit by NLL on a "
        "held-in split), split-conformal nonconformity (proper calibration split), "
        "deep-kNN distance in the ENCODER representation, Mahalanobis in the "
        "ENCODER representation. Compared against deployable misfit (test CE).\n"
    )
    L.append("\n| estimator | ADNI rho | OASIS rho |")
    L.append("|---|---|---|")
    panel = result["stronger_panel"]["per_dataset"]
    ests = sorted(
        {k for ds in panel for k in panel[ds]},
        key=lambda e: -panel.get("adni", {}).get(e, {}).get("mean", -9),
    )
    for e in ests:
        a_ = panel.get("adni", {}).get(e, {})
        o_ = panel.get("oasis", {}).get(e, {})
        am = f"{a_['mean']:+.3f}±{a_['std']:.3f}" if a_ else "--"
        om = f"{o_['mean']:+.3f}±{o_['std']:.3f}" if o_ else "--"
        if e == "structural_misfit_train":
            star = " (DEPLOYABLE misfit = anchor)"
        elif e == "structural_misfit_test":
            star = " (test-CE misfit, reference upper bound)"
        else:
            star = ""
        L.append(f"| {e}{star} | {am} | {om} |")
    L.append(
        "\nThe training-side misfit in the table is the row baselines must "
        "beat; test-CE misfit is an oracle reference that uses test labels.\n"
    )
    L.append(f"\n**Verdict (Block 2):** {result['stronger_panel']['verdict']}\n")

    # ---- Block 3 ----
    L.append("\n## Block 3 -- Clinical utility of misfit-based selective prediction\n")
    for ds, c in result["clinical_utility"]["per_dataset"].items():
        L.append(f"\n### {ds.upper()}\n")
        L.append(
            f"ECE = **{c['ece']:.3f}**; reliability monotonicity (bin vs empirical "
            f"error) rho = **{c['monotonicity_rho']:+.2f}** "
            f"(p={c['monotonicity_p']:.2g}); error prevalence = "
            f"{c['prevalence_error']:.3f}.\n"
        )
        L.append("\nReliability diagram (misfit-quantile bins):\n")
        L.append("| bin | misfit mean | misfit (norm) | empirical error | n |")
        L.append("|---|---|---|---|---|")
        for b in c["reliability_bins"]:
            L.append(
                f"| {b['bin']} | {b['misfit_mean']:.3f} | "
                f"{b['misfit_norm_mean']:.3f} | {b['empirical_error']:.3f} | "
                f"{b['n']} |"
            )
        L.append("\nCoverage-accuracy frontier (misfit ranking):\n")
        L.append("| coverage | accuracy |")
        L.append("|---|---|")
        for ck in ["100", "90", "80", "70", "60", "50", "40", "30"]:
            fr = c["coverage_accuracy"][ck]
            L.append(f"| {ck}% | {fr['acc_mean']*100:.1f}±{fr['acc_std']*100:.1f}% |")
        L.append("\nDecision curve / net benefit (abstain top-p_t by misfit):\n")
        L.append("| p_t | NB misfit | NB treat-all | NB treat-none | adv over default |")
        L.append("|---|---|---|---|---|")
        for nb in c["net_benefit_curve"]:
            L.append(
                f"| {nb['p_t']:.2f} | {nb['net_benefit_misfit']:+.3f} | "
                f"{nb['net_benefit_treat_all']:+.3f} | "
                f"{nb['net_benefit_treat_none']:.3f} | "
                f"{nb['advantage_over_best_default']:+.3f} |"
            )
        L.append(
            f"\nNet benefit >= best default at all thresholds: "
            f"**{c['net_benefit_positive_all_thresholds']}**.\n"
        )
    L.append(f"\n**Verdict (Block 3):** {result['clinical_utility']['verdict']}\n")

    (OUT / "enrichment.md").write_text("\n".join(L))


# ===========================================================================
# Main
# ===========================================================================
def main():
    """Run the three blocks on ADNI and OASIS-1; write enrichment.json and enrichment.md."""
    print(f"device={DEVICE}", flush=True)
    print("Loading datasets...", flush=True)
    adni = load_adni()
    oasis = load_oasis_ds()
    datasets = {"adni": adni, "oasis": oasis}

    result = {
        "meta": {
            "target": "per_subject_binary_error (1 if misclassified in held-out fold)",
            "protocol": "leave-one-environment-out, pooled across folds, 5 seeds",
            "n_seeds": N_SEEDS,
            "misfit_anchor": reporting_reference('misfit_anchor', required=False),
            "numerics": "float64 covariance; no torch.inverse (linalg.solve/SVD pinv); "
                        "condition-number guards (warn if kappa>1e6)",
        }
    }

    # ---- Block 1: shift severity ----
    print("\n########## BLOCK 1: SHIFT-SEVERITY ##########", flush=True)
    per_env = {}
    for ds, (feat, tgt, sites) in datasets.items():
        t0 = time.time()
        per_env[ds] = shift_severity(ds, feat, tgt, sites)
        print(f"  {ds}: {len(per_env[ds])} envs ({time.time()-t0:.0f}s)", flush=True)
    analysis = analyze_shift(per_env)
    result["shift_severity"] = {
        "per_env": per_env,
        "analysis": analysis,
        "verdict": verdict_shift(analysis),
    }
    print("  " + result["shift_severity"]["verdict"], flush=True)

    # ---- Block 2: stronger panel ----
    print("\n########## BLOCK 2: STRONGER BASELINE PANEL ##########", flush=True)
    panel = {}
    for ds, (feat, tgt, sites) in datasets.items():
        t0 = time.time()
        panel[ds] = stronger_panel(ds, feat, tgt, sites)
        print(f"  {ds} panel done ({time.time()-t0:.0f}s)", flush=True)
    result["stronger_panel"] = {
        "per_dataset": panel,
        "verdict": verdict_panel(panel),
    }
    print("  " + result["stronger_panel"]["verdict"], flush=True)

    # ---- Block 3: clinical utility ----
    print("\n########## BLOCK 3: CLINICAL UTILITY ##########", flush=True)
    clin = {}
    for ds, (feat, tgt, sites) in datasets.items():
        t0 = time.time()
        clin[ds] = clinical_utility(ds, feat, tgt, sites)
        a100 = clin[ds]["coverage_accuracy"]["100"]["acc_mean"] * 100
        a50 = clin[ds]["coverage_accuracy"]["50"]["acc_mean"] * 100
        print(
            f"  {ds}: ECE={clin[ds]['ece']:.3f} "
            f"cov-acc {a100:.1f}->{a50:.1f}@50% ({time.time()-t0:.0f}s)",
            flush=True,
        )
    result["clinical_utility"] = {
        "per_dataset": clin,
        "verdict": verdict_clinical(clin),
    }
    print("  " + result["clinical_utility"]["verdict"], flush=True)

    # ---- save ----
    (OUT / "enrichment.json").write_text(json.dumps(result, indent=2))
    write_md(result)
    print(f"\nSaved {OUT / 'enrichment.json'} and {OUT / 'enrichment.md'}", flush=True)

    print("\n===== VERDICT SUMMARY =====", flush=True)
    print("1. SHIFT-SEVERITY: " + result["shift_severity"]["verdict"], flush=True)
    print("2. STRONGER PANEL: " + result["stronger_panel"]["verdict"], flush=True)
    print("3. CLINICAL UTIL : " + result["clinical_utility"]["verdict"], flush=True)


if __name__ == "__main__":
    main()
