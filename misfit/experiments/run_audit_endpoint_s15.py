"""AT-S15 (design A): one independent development-audit endpoint.

Question: after a development audit (phase A: the paper's 5 seeds), does the cross-refit
training loss predict a subject's failure frequency in INDEPENDENTLY REPEATED later
evaluations (phase B: 5 fresh seeds, same LOO protocol) beyond (i) the phase-A observed
error flag (the known failures a reviewer would inspect anyway) and (ii) ordinary
single-fit training loss?

Reads results/perseed_misfit_two_phases.npz. Analyses per cohort:
  1. predictors of err_B (phase-B failure frequency): misfit_A, errflag_A (fraction of
     phase-A evaluations wrong), ordinary_loss (one all-data PlainMLP fit, seed 0).
  2. incremental value: partial Spearman of misfit_A vs err_B controlling errflag_A
     (and controlling both errflag_A + ordinary loss), with B=2000 subject-bootstrap CIs.
  3. the audit case that matters: among subjects with NO phase-A error (errflag_A == 0),
     AUROC of misfit_A (and of ordinary loss) for predicting any phase-B failure.
CPU except the single ordinary-loss fits. Output: results/audit_endpoint_s15.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import sys

# CUDA visibility is selected by the caller.
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    PlainMLP, load_adni, load_oasis_ds)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NPZ = _resolve_path("<REPO_ROOT>/misfit/results/perseed_misfit_two_phases.npz")
OUT = _resolve_path("<REPO_ROOT>/misfit/results/audit_endpoint_s15.json")
B = 2000


def ordinary_loss(X, y):
    """Per-subject training cross-entropy of a single all-data PlainMLP fit (seed 0)."""
    torch.manual_seed(0)
    m = PlainMLP(X.shape[1], 2).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = X.to(DEV), y.long().to(DEV)
    m.train()
    for _ in range(100):
        opt.zero_grad()
        F.cross_entropy(m(Xt), yt).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return F.cross_entropy(m(Xt), yt, reduction="none").cpu().numpy()


def partial_spearman(x, target, controls):
    """Spearman of rank-residuals of x and target after regressing out control ranks."""
    rx, rt = rankdata(x), rankdata(target)
    C = np.column_stack([rankdata(c) for c in controls] + [np.ones(len(x))])
    bx = np.linalg.lstsq(C, rx, rcond=None)[0]
    bt = np.linalg.lstsq(C, rt, rcond=None)[0]
    ex, et = rx - C @ bx, rt - C @ bt
    return float(spearmanr(ex, et)[0])


def boot_ci(fn, n, rng, B=B):
    """95% percentile bootstrap CI of fn over B subject resamples; non-finite draws dropped."""
    vals = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def main():
    """Run the S15 audit-endpoint analysis per cohort and write the JSON output."""
    d = np.load(NPZ)
    res = {}
    for name, loader in [("adni", load_adni), ("oasis", load_oasis_ds)]:
        X, y, _ = loader()
        misA = d[f"{name}_mis_A"].mean(0)
        errA = d[f"{name}_err_A"].mean(0)
        errB = d[f"{name}_err_B"].mean(0)
        ordl = ordinary_loss(X, y)
        rng = np.random.RandomState(7)
        r = {"n": int(len(misA)),
             "testretest_errA_vs_errB": round(float(spearmanr(errA, errB)[0]), 3),
             "rho_misfitA_errB": round(float(spearmanr(misA, errB)[0]), 3),
             "rho_errflagA_errB": round(float(spearmanr(errA, errB)[0]), 3),
             "rho_ordinary_errB": round(float(spearmanr(ordl, errB)[0]), 3)}
        pc1 = partial_spearman(misA, errB, [errA])
        pc2 = partial_spearman(misA, errB, [errA, ordl])
        r["partial_misfit_vs_errB_given_errflag"] = {
            "rho": round(pc1, 3),
            "ci95": boot_ci(lambda i: partial_spearman(misA[i], errB[i], [errA[i]]),
                            len(misA), rng)}
        r["partial_misfit_vs_errB_given_errflag_and_ordinary"] = {
            "rho": round(pc2, 3),
            "ci95": boot_ci(lambda i: partial_spearman(misA[i], errB[i], [errA[i], ordl[i]]),
                            len(misA), rng)}
        # zero-flag subset: subjects with no phase-A error
        z = errA == 0
        evt = (errB[z] > 0).astype(int)
        r["zeroflag_subset"] = {"n": int(z.sum()), "n_future_failures": int(evt.sum())}
        if 0 < evt.sum() < len(evt):
            r["zeroflag_subset"]["auroc_misfit"] = round(float(roc_auc_score(evt, misA[z])), 3)
            r["zeroflag_subset"]["auroc_ordinary"] = round(float(roc_auc_score(evt, ordl[z])), 3)
            zi = np.where(z)[0]
            r["zeroflag_subset"]["auroc_misfit_ci95"] = boot_ci(
                lambda i: roc_auc_score((errB[zi[i]] > 0).astype(int), misA[zi[i]])
                if 0 < (errB[zi[i]] > 0).sum() < len(i) else np.nan, len(zi), rng)
        res[name] = r
        print(name, json.dumps(r, indent=1), flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
