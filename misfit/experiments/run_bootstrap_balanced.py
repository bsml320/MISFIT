"""Reviewer-requested additions: (H6) subject-level bootstrap 95% CIs on the headline
rho(misfit,error) and failure-detection AUROC (the 5-seed std only captures init variance),
and (H5) class-BALANCED accepted-set accuracy vs coverage with the majority-class floor
(the raw selective gain is partly prevalence-driven). ADNI + OASIS-1, 5 seeds. GPU 2.
Output: results/bootstrap_balanced.json.
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
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    N_SEEDS, SEED_BASE, load_adni, load_oasis_ds)
from misfit.experiments.run_clinical_stratified_media import collect  # noqa: E402

RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
COVS = [1.0, 0.8, 0.5, 0.3]
RNG = np.random.RandomState(0)


def per_subject(loader):
    """Collect seed-averaged per-subject misfit and error for one cohort.

    Returns:
        Tuple (misfit mean over seeds, error fraction over seeds, binary error
        at the 0.5 threshold, true labels, seed-0 predictions), all restricted
        to tested subjects.
    """
    f, t, s = loader()
    mis, err, pred, true = [], [], [], []
    for k in range(N_SEEDS):
        c = collect(f, t, s, SEED_BASE + k)
        m = c["tested"]
        mis.append(c["misfit"][m]); err.append(c["err"][m])
        pred.append(c["pred"][m]); true.append(c["true"][m])
    # seed-averaged per subject (tested mask identical across seeds)
    mis = np.mean(mis, 0); err_frac = np.mean(err, 0)
    err_bin = (err_frac >= 0.5).astype(int)
    return mis, err_frac, err_bin, np.array(true[0]), np.array(pred[0])


def boot_ci(mis, err_bin, err_frac, B=2000):
    """Subject-level 95% percentile bootstrap CIs for rho and AUROC (B resamples)."""
    n = len(mis); rhos, aucs = [], []
    for _ in range(B):
        idx = RNG.randint(0, n, n)
        e = err_bin[idx]
        if 0 < e.sum() < len(e):
            aucs.append(roc_auc_score(e, mis[idx]))
        r = spearmanr(mis[idx], err_frac[idx])[0]
        if not np.isnan(r):
            rhos.append(r)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"rho_ci": q(rhos), "auroc_ci": q(aucs)}


def balanced_frontier(mis, true, pred):
    """Accepted-set metrics at each COVS coverage, lowest-misfit subjects first.

    Each row holds raw accuracy, class-balanced accuracy (mean of sensitivity
    and specificity), and the majority-class accuracy floor of the accepted set.
    """
    order = np.argsort(mis)
    rows = []
    for cov in COVS:
        n = max(1, int(round(cov * len(mis))))
        a = order[:n]
        t, p = true[a], pred[a]
        pos, neg = t == 1, t == 0
        sens = (p[pos] == 1).mean() if pos.sum() else float("nan")
        spec = (p[neg] == 0).mean() if neg.sum() else float("nan")
        bal = float(np.nanmean([sens, spec]))
        maj = float(max(t.mean(), 1 - t.mean()))          # majority-class accuracy on accepted set
        acc = float((p == t).mean())
        rows.append({"coverage": cov, "accuracy": acc, "balanced_accuracy": bal,
                     "majority_floor": maj, "sens": float(sens), "spec": float(spec)})
    return rows


def main():
    """Run both cohorts and write results/bootstrap_balanced.json."""
    out = {}
    for name, loader in [("adni", load_adni), ("oasis", load_oasis_ds)]:
        mis, err_frac, err_bin, true, pred = per_subject(loader)
        auc = roc_auc_score(err_bin, mis)
        rho = spearmanr(mis, err_frac)[0]
        ci = boot_ci(mis, err_bin, err_frac)
        fr = balanced_frontier(mis, true, pred)
        out[name] = {"n": int(len(mis)), "auroc": float(auc), "rho": float(rho),
                     **ci, "balanced_frontier": fr}
        print(f"[{name}] N={len(mis)} AUROC={auc:.3f} CI{ci['auroc_ci']} "
              f"rho={rho:.3f} CI{ci['rho_ci']}", flush=True)
        for r in fr:
            print(f"   cov {int(r['coverage']*100)}%: acc {r['accuracy']:.3f} "
                  f"bal-acc {r['balanced_accuracy']:.3f} (floor {r['majority_floor']:.3f})", flush=True)
    (RES / "bootstrap_balanced.json").write_text(json.dumps(out, indent=2))
    print("wrote results/bootstrap_balanced.json")


if __name__ == "__main__":
    main()
