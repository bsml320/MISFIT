"""Attempts to FIX two stated limitations of the MedIA reliability paper.

The paper flags two genuinely addressable limitations (the rest are inherent:
seed-unstable loadings, no external cohort without a DUA, clinical operating
point). Both get a concrete attempt here; whatever survives honest evaluation
is kept.

FIX 1 -- deployment gap. The retrospective misfit needs the subject to have
been in training; the label-free SNT proxy retains only part of the signal on
a novel subject. The limitation names the untried remedy: "a model that
generalizes the structural-misfit signal to unlabeled subjects." A LEARNED
CROSS-SITE FAILURE PREDICTOR is trained on subjects from the OTHER environments
(their held-out errors are observed = past labeled sites) and applied to the
target environment by FEATURES ONLY (deployable, no target labels). Compared
against the SNT kNN-misfit baseline. Estimators: logistic regression, gradient
boosting, random forest on the raw features.

FIX 2 -- calibration. Misfit is a ranking score, not a calibrated probability.
Misfit is mapped to P(error) with isotonic and Platt scaling on a calibration
split, and ECE is measured on a disjoint test split, under both exchangeable
(random) and shifted (leave-one-protocol-out) calibration.

Reuses the validated collector from run_ml_reliability_media (binary per-subject
error target, leave-one-environment-out, 5 seeds). Output:
misfit/results/limitation_fixes.{json,md}
"""

from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))

from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    N_SEEDS,
    SEED_BASE,
    _rho,
    load_adni,
    load_oasis_ds,
)
from misfit.experiments.run_ml_reliability_media import (  # noqa: E402
    collect_seed,
)

warnings.filterwarnings("ignore")

OUT = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
SNT_K = 10
N_ECE_BINS = 10


# ===========================================================================
# FIX 1 -- learned cross-site failure predictor vs SNT
# ===========================================================================
def _snt_scores(X, misfit, tr_mask, te_mask):
    """SNT proxy: mean misfit of k nearest TRAIN-env subjects, for te_mask subjects."""
    nn = NearestNeighbors(n_neighbors=min(SNT_K, int(tr_mask.sum()))).fit(X[tr_mask])
    _, idx = nn.kneighbors(X[te_mask])
    return misfit[tr_mask][idx].mean(1)


def block_deployment(data_by_seed, features):
    """FIX 1: leave-one-site-out failure prediction from features only.

    For each target site, fits each estimator on the other sites' (features,
    held-out error) pairs and scores the target site from features alone; the
    SNT baseline instead transfers mean kNN training misfit. Reported as
    Spearman rho against held-out error, per seed.
    """
    X = features.numpy().astype(np.float64)
    estimators = {
        "snt_knn_misfit": None,  # special-cased
        "logreg": lambda: LogisticRegression(max_iter=500, C=1.0),
        "grad_boost": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=6),
    }
    res = {}
    for name, ctor in estimators.items():
        rhos = []
        for d in data_by_seed:
            t = d["tested"]; err = d["err"]; site = d["site"]; misfit = d["misfit"]
            preds = np.full(len(X), np.nan)
            for tgt in np.unique(site[t]):
                tr = t & (site != tgt) & ~np.isnan(err)
                te = t & (site == tgt)
                if tr.sum() < 20 or te.sum() < 5:
                    continue
                if err[tr].sum() in (0, tr.sum()):
                    continue
                if name == "snt_knn_misfit":
                    tr_snt = t & (site != tgt) & ~np.isnan(misfit)
                    preds[te] = _snt_scores(X, misfit, tr_snt, te)
                else:
                    clf = ctor().fit(X[tr], err[tr].astype(int))
                    preds[te] = clf.predict_proba(X[te])[:, 1]
            m = t & ~np.isnan(preds) & ~np.isnan(err)
            r = _rho(preds[m], err[m])
            if not np.isnan(r):
                rhos.append(r)
        res[name] = {"rho_mean": float(np.mean(rhos)) if rhos else None,
                     "rho_std": float(np.std(rhos)) if rhos else None,
                     "per_seed": [round(float(x), 3) for x in rhos]}
    return res


# ===========================================================================
# FIX 2 -- calibrate misfit -> P(error); ECE before/after
# ===========================================================================
def _ece(p, y, bins=N_ECE_BINS):
    """Expected calibration error with equal-width bins; last bin right-closed."""
    p = np.clip(p, 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for b in range(bins):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < bins - 1 else p <= edges[b + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(e)


def block_calibration(data_by_seed):
    """FIX 2: ECE of raw min-max-scaled misfit vs isotonic/Platt-calibrated
    P(error), under exchangeable (random 50/50 split, 10 repeats) and shifted
    (leave-one-protocol-out) calibration."""
    rng = np.random.RandomState(0)
    out = {"raw": [], "isotonic_exch": [], "platt_exch": [],
           "isotonic_shift": [], "platt_shift": []}
    for d in data_by_seed:
        t = d["tested"]; mis = d["misfit"][t]; err = d["err"][t]; site = d["site"][t]
        mn = (mis - mis.min()) / (mis.max() - mis.min() + 1e-12)  # raw "prob"
        out["raw"].append(_ece(mn, err))
        # exchangeable: random 50/50 cal/test, 10 repeats
        for _ in range(10):
            idx = rng.permutation(len(mis)); h = len(mis) // 2
            cal, te = idx[:h], idx[h:]
            if len(np.unique(err[cal])) < 2:
                continue  # Platt/logistic needs both error classes in the calibration split
            iso = IsotonicRegression(out_of_bounds="clip").fit(mis[cal], err[cal])
            out["isotonic_exch"].append(_ece(iso.predict(mis[te]), err[te]))
            pl = LogisticRegression(max_iter=500).fit(mis[cal, None], err[cal].astype(int))
            out["platt_exch"].append(_ece(pl.predict_proba(mis[te, None])[:, 1], err[te]))
        # shifted: leave-one-protocol-out calibration
        for p in np.unique(site):
            cal = site != p; te = site == p
            if te.sum() < 5 or cal.sum() < 20:
                continue
            if len(np.unique(err[cal])) < 2:
                continue  # Platt/logistic needs both error classes in the calibration split
            iso = IsotonicRegression(out_of_bounds="clip").fit(mis[cal], err[cal])
            out["isotonic_shift"].append(_ece(iso.predict(mis[te]), err[te]))
            pl = LogisticRegression(max_iter=500).fit(mis[cal, None], err[cal].astype(int))
            out["platt_shift"].append(_ece(pl.predict_proba(mis[te, None])[:, 1], err[te]))
    return {k: (float(np.mean(v)) if v else None) for k, v in out.items()}


def run_dataset(name, loader):
    """Collect per-seed data once and run both fix blocks on one cohort."""
    t0 = time.time()
    features, targets, site_ids = loader()
    print(f"[{name}] N={features.shape[0]} d={features.shape[1]} "
          f"E={int(site_ids.max())+1}", flush=True)
    data_by_seed = [collect_seed(features, targets, site_ids, SEED_BASE + s)
                    for s in range(N_SEEDS)]
    sanity = np.mean([_rho(d["misfit"][d["tested"]], d["err"][d["tested"]])
                      for d in data_by_seed])
    print(f"  [{name}] misfit rho sanity={sanity:.3f}", flush=True)
    dep = block_deployment(data_by_seed, features)
    print(f"  [{name}] FIX1 deployment: " +
          ", ".join(f"{k}={v['rho_mean']:.3f}" for k, v in dep.items()
                    if v['rho_mean'] is not None), flush=True)
    cal = block_calibration(data_by_seed)
    print(f"  [{name}] FIX2 calibration ECE: raw={cal['raw']:.3f} "
          f"iso_exch={cal['isotonic_exch']:.3f} iso_shift={cal['isotonic_shift']:.3f}",
          flush=True)
    return {"n": int(features.shape[0]),
            "misfit_rho_sanity": float(sanity),
            "fix1_deployment": dep, "fix2_calibration_ece": cal,
            "wall_s": round(time.time() - t0, 1)}


def main():
    result = {}
    result["adni"] = run_dataset("adni", load_adni)
    result["oasis"] = run_dataset("oasis", load_oasis_ds)
    (OUT / "limitation_fixes.json").write_text(json.dumps(result, indent=2))
    L = ["# Limitation fixes for the MedIA paper", "",
         "FIX 1: learned cross-site failure predictor (deployable, features-only) vs the "
         "SNT kNN-misfit proxy (~0.32). FIX 2: calibrate misfit -> P(error). "
         f"Reuses the validated collector; {N_SEEDS} seeds.", ""]
    for name in ["adni", "oasis"]:
        r = result[name]
        L += [f"## {name.upper()} (N={r['n']}, misfit rho sanity={r['misfit_rho_sanity']:.3f})",
              "", "### FIX 1 -- deployment: rho(score, error) on a held-out site, features only",
              "| estimator | rho | per-seed |", "|---|---|---|"]
        for k, v in r["fix1_deployment"].items():
            rm = f"{v['rho_mean']:.3f}+/-{v['rho_std']:.3f}" if v["rho_mean"] is not None else "--"
            L.append(f"| {k} | {rm} | {v['per_seed']} |")
        c = r["fix2_calibration_ece"]
        L += ["", "### FIX 2 -- calibration ECE (lower = better)",
              "| method | ECE |", "|---|---|"]
        for k in ["raw", "isotonic_exch", "platt_exch", "isotonic_shift", "platt_shift"]:
            v = c[k]
            L.append(f"| {k} | {v:.3f} |" if v is not None else f"| {k} | -- |")
        L.append("")
    (OUT / "limitation_fixes.md").write_text("\n".join(L))
    print(f"\nSaved {OUT / 'limitation_fixes.json'}", flush=True)


if __name__ == "__main__":
    main()
