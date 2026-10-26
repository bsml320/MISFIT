"""AT-S16: joint class-by-environment analysis.

Separate within-class and within-environment analyses do not establish the association
within their intersection. Here, per (diagnostic class c, held-out environment e)
stratum: subject counts, error/nonerror counts, and the misfit-error association
computed per phase-A seed inside the stratum (Spearman vs the seed's binary error;
AUROC where both outcomes present and n>=8). Aggregation rule (declared): equal weight
per estimable stratum-seed cell; mean +/- SD over cells, plus the fraction of positive
cells. Strata with a single outcome class in a seed are excluded from that seed and
counted as non-estimable. Reads results/perseed_misfit_two_phases.npz. CPU.
Output: results/class_env_strata_s16.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

NPZ = _resolve_path("<REPO_ROOT>/misfit/results/perseed_misfit_two_phases.npz")
OUT = _resolve_path("<REPO_ROOT>/misfit/results/class_env_strata_s16.json")
MIN_N = 8

d = np.load(NPZ)
res = {}
for name in ("adni", "oasis"):
    site = d[f"{name}_site"]
    lab = d[f"{name}_label"]
    misS = d[f"{name}_mis_A"]      # (5, N)
    errS = d[f"{name}_err_A"]      # (5, N) binary per seed
    strata = []
    cell_rhos, cell_aucs = [], []
    for c in (0, 1):
        for e in np.unique(site):
            m = (lab == c) & (site == e)
            n = int(m.sum())
            nerr = float(errS[:, m].mean(0).sum())  # expected errors (seed-mean)
            row = {"class": int(c), "env": int(e), "n": n,
                   "mean_errors": round(nerr, 1), "estimable_seeds": 0}
            rhos, aucs = [], []
            for s in range(misS.shape[0]):
                ee = errS[s, m]
                if n < MIN_N or ee.sum() == 0 or ee.sum() == n:
                    continue
                r = spearmanr(misS[s, m], ee)[0]
                if np.isfinite(r):
                    rhos.append(r)
                    aucs.append(roc_auc_score(ee, misS[s, m]))
            row["estimable_seeds"] = len(rhos)
            if rhos:
                row["rho_mean"] = round(float(np.mean(rhos)), 3)
                row["rho_sd"] = round(float(np.std(rhos)), 3)
                row["auroc_mean"] = round(float(np.mean(aucs)), 3)
                cell_rhos += rhos
                cell_aucs += aucs
            strata.append(row)
    est = [r for r in strata if r["estimable_seeds"] > 0]
    res[name] = {
        "strata": strata,
        "n_strata_total": len(strata),
        "n_strata_estimable": len(est),
        "pooled_over_cells": {
            "rho_mean": round(float(np.mean(cell_rhos)), 3),
            "rho_sd": round(float(np.std(cell_rhos)), 3),
            "auroc_mean": round(float(np.mean(cell_aucs)), 3),
            "frac_cells_positive": round(float(np.mean([r > 0 for r in cell_rhos])), 3),
            "n_cells": len(cell_rhos)},
        "rule": "equal weight per estimable stratum-seed cell; single-outcome or n<8 "
                "cells non-estimable; per-seed binary error",
    }
    p = res[name]["pooled_over_cells"]
    print(f"[{name}] estimable strata {len(est)}/{len(strata)}; cells n={p['n_cells']} "
          f"rho={p['rho_mean']}±{p['rho_sd']} AUROC={p['auroc_mean']} "
          f"{p['frac_cells_positive']*100:.0f}% positive", flush=True)
json.dump(res, open(OUT, "w"), indent=2)
print("wrote", OUT)
