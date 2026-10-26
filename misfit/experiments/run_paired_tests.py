"""Paired significance tests (reviewer): is structural misfit statistically distinguishable from
the best label-using training-dynamics scores? Paired subject-level bootstrap of the AUROC and rho
DIFFERENCE (misfit minus competitor); a 95% CI containing 0 means a statistical tie. ADNI+OASIS,
pooled over 5 seeds. Participant-ID resampling (AT-S10). Output: results/paired_tests.json.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path
import json, os, sys
from pathlib import Path
# CUDA visibility is selected by the caller.
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
sys.path.insert(0, _resolve_path("<REPO_ROOT>/misfit/experiments"))
from run_training_dynamics import collect_dynamics, SCORES  # noqa: E402
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    N_SEEDS, SEED_BASE, load_adni, load_oasis_ds)
RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
RNG = np.random.RandomState(0)
COMPARE = ["aum_neg", "el2n", "forgetting"]


def paired(mis, comp, err, B=2000, n_subj=None):
    """Participant-aware paired bootstrap: arrays are seed-major concatenations of
    n_subj participants x N_SEEDS rows; resample PARTICIPANT ids and carry every
    seed row of a drawn participant for both scores together."""
    n = len(err); d_auc, d_rho = [], []
    n_seeds = n // n_subj
    base = np.arange(n_seeds)[:, None] * n_subj
    for _ in range(B):
        subj = RNG.randint(0, n_subj, n_subj)
        idx = (base + subj[None, :]).ravel(); e = err[idx]
        if 0 < e.sum() < len(e):
            d_auc.append(roc_auc_score(e, mis[idx]) - roc_auc_score(e, comp[idx]))
        rm = spearmanr(mis[idx], e)[0]; rc = spearmanr(comp[idx], e)[0]
        if not (np.isnan(rm) or np.isnan(rc)):
            d_rho.append(rm - rc)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    tie = lambda a: bool(np.percentile(a, 2.5) <= 0 <= np.percentile(a, 97.5))
    return {"dAUROC_ci": ci(d_auc), "dAUROC_tie": tie(d_auc),
            "dRho_ci": ci(d_rho), "dRho_tie": tie(d_rho)}


def main():
    """Collect scores per seed, concatenate seed-major, run paired comparisons."""
    RES.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, loader in [("adni", load_adni), ("oasis", load_oasis_ds)]:
        f, t, s = loader()
        allsc = {k: [] for k in SCORES}; allerr = []
        for seed in range(N_SEEDS):
            sc, err, m = collect_dynamics(f, t, s, SEED_BASE + seed)
            for k in SCORES:
                allsc[k].append(sc[k][m])
            allerr.append(err[m])
        sc = {k: np.concatenate(allsc[k]) for k in SCORES}; err = np.concatenate(allerr)
        out[name] = {c: paired(sc["misfit"], sc[c], err, n_subj=len(allerr[0])) for c in COMPARE}
        print(f"[{name}] misfit vs (paired subject-bootstrap, tie=CI contains 0):")
        for c in COMPARE:
            r = out[name][c]
            print(f"   vs {c:11s} dAUROC {r['dAUROC_ci']} tie={r['dAUROC_tie']} | "
                  f"dRho {r['dRho_ci']} tie={r['dRho_tie']}", flush=True)
    (RES / "paired_tests.json").write_text(json.dumps(out, indent=2))
    print("wrote results/paired_tests.json")


if __name__ == "__main__":
    main()
