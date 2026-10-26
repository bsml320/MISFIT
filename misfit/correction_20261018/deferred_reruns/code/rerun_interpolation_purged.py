"""R1: corrected ADNI arm of the training-duration (interpolation) sweep.

Line-faithful rerun of misfit/experiments/run_interpolation_ablation.py
(ADNI arm) with exactly the two plan deltas (RERUN_PLAN.md Section 2):
  D1. train mask per fold = purge_masks (participant-purged LOSO);
  D2. per-fold source-only features (fold_pca on the purged training pool),
      computed once per fold and shared across seeds, instead of the shared
      cohort matrix.
Unchanged: PlainMLP [64,32], Adam lr 1e-3, weight decay 0, full batch, 3000
epochs, checkpoints [25, 50, 100, 200, 400, 800, 1600, 3000], seeds 42-44,
score = per-acquisition training CE at each checkpoint averaged over the folds
in which the acquisition trains, target = held-out error of the final
(3000-epoch) model, pooled Spearman and AUROC over tested acquisitions,
np.std ddof=0. GPU 2.

Output: deferred_reruns/results/interpolation_purged.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import sys
from pathlib import Path

# CUDA visibility is selected by the caller.
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerun_common import load, purged_fold_projections  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    DEVICE, PlainMLP)

CKPTS = [25, 50, 100, 200, 400, 800, 1600, 3000]
SEEDS, SEED_BASE = 3, 42
RES = Path(__file__).resolve().parents[1] / "results" / "interpolation_purged.json"


def run_seed(folds, targets, seed):
    """One seed of the purged training-duration sweep.

    Per fold: train PlainMLP to the last checkpoint, snapshot per-acquisition
    training CE at every checkpoint, and keep the final model's held-out
    predictions. Checkpoint scores average over the folds in which an
    acquisition trains.

    Returns:
        Dict checkpoint -> {rho, auroc, mean_train_loss} over tested acquisitions.
    """
    n = len(targets)
    ssum = {ep: np.zeros(n) for ep in CKPTS}
    cnt = np.zeros(n)
    pred = np.full(n, -1)
    tested = np.zeros(n, bool)
    mtl = {ep: [] for ep in CKPTS}
    y = targets.long().to(DEVICE)
    for held in sorted(folds):
        f = folds[held]
        idx = np.where(f["train"])[0]
        te = np.where(f["test"])[0]
        Z = f["Z"].to(DEVICE)
        torch.manual_seed(seed)
        m = PlainMLP(Z.shape[1], 2).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=0.0)
        Xt, yt = Z[idx], y[idx]
        m.train()
        for ep in range(1, max(CKPTS) + 1):
            opt.zero_grad()
            F.cross_entropy(m(Xt), yt).backward()
            opt.step()
            if ep in CKPTS:
                # snapshot per-acquisition training CE at this checkpoint
                with torch.no_grad():
                    losses = F.cross_entropy(m(Xt), yt, reduction="none").cpu().numpy()
                ssum[ep][idx] += losses
                mtl[ep].append(float(losses.mean()))
        cnt[idx] += 1
        m.eval()
        with torch.no_grad():
            pred[te] = m(Z[te]).argmax(1).cpu().numpy()
        tested[te] = True
    err = (pred != targets.numpy().astype(int)).astype(float)
    assert (cnt[tested] > 0).all(), "a tested acquisition has no training refit"
    out = {}
    for ep in CKPTS:
        sc = (ssum[ep] / np.maximum(cnt, 1.0))[tested]
        e = err[tested]
        rho = float(spearmanr(sc, e)[0])
        auc = float(roc_auc_score(e, sc)) if 0 < e.sum() < len(e) else float("nan")
        out[ep] = {"rho": rho, "auroc": auc, "mean_train_loss": float(np.mean(mtl[ep]))}
    return out


def main():
    """Run the 3-seed sweep over the purged folds and write the aggregate JSON."""
    d = load()
    folds = purged_fold_projections(d["raw"], d["site_ids"], d["ptids"])
    assert len(folds) == 8, f"fold set changed: {sorted(folds)} (plan C9)"
    targets = d["targets"]
    print(f"[adni purged] N={len(targets)} folds={sorted(folds)}", flush=True)
    per = {ep: {"rho": [], "auroc": [], "mtl": []} for ep in CKPTS}
    for k in range(SEEDS):
        r = run_seed(folds, targets, SEED_BASE + k)
        for ep in CKPTS:
            per[ep]["rho"].append(r[ep]["rho"])
            per[ep]["auroc"].append(r[ep]["auroc"])
            per[ep]["mtl"].append(r[ep]["mean_train_loss"])
        print(f"  seed {k} done", flush=True)
    res = {"adni_purged": {str(ep): {
        "rho_mean": float(np.mean(v["rho"])), "rho_std": float(np.std(v["rho"])),
        "auroc_mean": float(np.nanmean(v["auroc"])), "auroc_std": float(np.nanstd(v["auroc"])),
        "mean_train_loss": float(np.mean(v["mtl"]))} for ep, v in per.items()}}
    for ep in CKPTS:
        r = res["adni_purged"][str(ep)]
        print(f"  ep{ep:5d}: train_loss={r['mean_train_loss']:.5f} "
              f"rho={r['rho_mean']:.3f}±{r['rho_std']:.3f} AUROC={r['auroc_mean']:.3f}",
              flush=True)
    json.dump(res, open(RES, "w"), indent=2)
    print("wrote", RES)


if __name__ == "__main__":
    main()
