"""Controlled synthetic that DISSOCIATES the two kinds of atypicality (reviewer H3/H4).

Subjects are generated with two INDEPENDENT difficulty axes:
  - label-conditional difficulty |s| : closeness to the true decision boundary (small |s| -> the
    label is ambiguous given the input -> the subject is genuinely error-prone).
  - feature-space atypicality ||u|| : magnitude of a label-IRRELEVANT nuisance block (an outlier in
    feature space that says nothing about the label).
They are drawn independently. At test time the nuisance block is SHIFTED (an OOD covariate shift),
so feature-density detectors fire but OOD error is still driven by |s|, not ||u||.

Hypothesis under test: structural misfit (per-subject training loss) tracks label-conditional
difficulty and hence OOD error, while Mahalanobis feature-density tracks nuisance atypicality and is
uninformative about error -- i.e. reliability under shift is label-conditional misfit, not
distributional outlierness. The design separates "training loss predicts error" from a tautology by
testing the mechanism.
GPU 2, 5 seeds. Output: results/synthetic_dissociation.json.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path
import json, os
from pathlib import Path
# CUDA visibility is selected by the caller.
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_TR, N_TE, D_NUIS, EPOCHS = 4000, 2000, 20, 200


def make(rng, n, nuisance_scale):
    """Sample n subjects with independent difficulty axes.

    A 1D signal s drives the label through a logistic flip of 3*s; a D_NUIS
    nuisance block u at the given scale is appended and never touches the label.

    Returns:
        (x, y, label_difficulty, atypicality) with label_difficulty = -|s|
        (higher = closer to the boundary) and atypicality = ||u||.
    """
    s = rng.normal(0, 1, n)                               # signal -> label
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-3 * s))).astype(int)  # label noise near s=0
    u = rng.normal(0, nuisance_scale, size=(n, D_NUIS))      # label-irrelevant nuisance
    x = np.concatenate([s[:, None] + rng.normal(0, 0.3, (n, 1)), u], axis=1)
    label_diff = -np.abs(s)                                  # high => harder (near boundary)
    atypicality = np.linalg.norm(u, axis=1)                  # feature-space outlierness
    return x.astype(np.float32), y, label_diff, atypicality


def run_seed(seed):
    """Train in-distribution, evaluate on nuisance-shifted test data.

    Computes test-time misfit (per-example CE against the true label) and a
    train-fit Mahalanobis distance in the penultimate representation, then
    correlates both with error and with the two generative difficulty axes.
    """
    rng = np.random.RandomState(seed)
    Xtr, ytr, _, _ = make(rng, N_TR, 1.0)
    Xte, yte, ldiff, atyp = make(rng, N_TE, 2.0)            # OOD: nuisance scaled up
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, 2)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(Xtr).to(DEV), torch.tensor(ytr).long().to(DEV)
    for _ in range(EPOCHS):
        opt.zero_grad(); F.cross_entropy(net(Xt), yt).backward(); opt.step()
    with torch.no_grad():
        Xv = torch.tensor(Xte).to(DEV)
        logits = net(Xv)
        pred = logits.argmax(1).cpu().numpy()
        misfit = F.cross_entropy(logits, torch.tensor(yte).long().to(DEV), reduction="none").cpu().numpy()
        rep = net[:4](Xv).cpu().numpy()                     # penultimate representation
        rep_tr = net[:4](Xt).cpu().numpy()
    err = (pred != yte).astype(float)
    mu = rep_tr.mean(0); cov = np.cov(rep_tr.T) + 1e-3 * np.eye(rep_tr.shape[1])
    ic = np.linalg.inv(cov); dif = rep - mu
    maha = np.einsum("ij,jk,ik->i", dif, ic, dif)           # feature-space density score
    def rho(a, b): return float(spearmanr(a, b)[0])
    def auc(a):
        return float(roc_auc_score(err, a)) if 0 < err.sum() < len(err) else float("nan")
    return {
        "misfit_vs_error_rho": rho(misfit, err), "misfit_vs_error_auroc": auc(misfit),
        "maha_vs_error_rho": rho(maha, err), "maha_vs_error_auroc": auc(maha),
        "misfit_vs_labeldiff_rho": rho(misfit, ldiff), "misfit_vs_atypicality_rho": rho(misfit, atyp),
        "maha_vs_labeldiff_rho": rho(maha, ldiff), "maha_vs_atypicality_rho": rho(maha, atyp),
        "base_error": float(err.mean()),
    }


def main():
    RES.mkdir(parents=True, exist_ok=True)
    keys = list(run_seed(0).keys())  # throwaway run only to enumerate metric keys
    runs = [run_seed(42 + s) for s in range(5)]
    agg = {k: {"mean": float(np.nanmean([r[k] for r in runs])),
               "std": float(np.nanstd([r[k] for r in runs]))} for k in keys}
    (RES / "synthetic_dissociation.json").write_text(json.dumps(agg, indent=2))
    print("=== synthetic dissociation (5 seeds, mean+/-std) ===")
    for k in keys:
        print(f"  {k:28s} {agg[k]['mean']:.3f} +/- {agg[k]['std']:.3f}")
    print("wrote results/synthetic_dissociation.json")


if __name__ == "__main__":
    main()
