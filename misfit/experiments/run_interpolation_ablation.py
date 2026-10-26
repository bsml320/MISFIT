"""Interpolation ablation (reviewer preempt): the reliability score is the FINAL per-subject
training loss, so what happens when the classifier is trained to near-interpolation?

Same MLP as the paper ([64,32]), trained 3000 epochs with NO weight decay (vs the paper's
100 epochs + wd 1e-4), full batch. At checkpoint epochs every training subject's CE is
recorded, aggregated across leave-one-environment-out folds, and correlated with held-out error.
This traces the ranking from the paper's regime into the interpolating regime.
ADNI + OASIS-3 (leakage-free, one session per participant). GPU 2, 3 seeds.
Output: results/interpolation_ablation.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import re
import sys

# CUDA visibility is selected by the caller.
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    DEVICE, PlainMLP, load_adni)

CKPTS = [25, 50, 100, 200, 400, 800, 1600, 3000]
SEEDS, SEED_BASE = 3, 42
RES = _resolve_path("<REPO_ROOT>/misfit/results/interpolation_ablation.json")


def load_oasis3_unique():
    """Load leakage-free OASIS-3: one earliest session per participant.

    Deduplicates by the _d<day> session suffix (smallest day wins) and codes
    scanners as contiguous integers ordered by descending frequency.

    Returns:
        (features, targets, site_ids) tensors aligned to the kept sessions.
    """
    d = np.load(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz"), allow_pickle=True)
    subj = d["subjects"].astype(str)
    sess = d["sessions"].astype(str)
    day = np.array([int(re.search(r"_d(\d+)$", s).group(1)) for s in sess])
    keep = np.array(sorted(np.where(subj == u)[0][np.argmin(day[subj == u])]
                           for u in np.unique(subj)))
    f = torch.tensor(d["features"][keep], dtype=torch.float32)
    t = torch.tensor(d["label"].astype(int)[keep])
    scan = d["scanner"].astype(str)[keep]
    uniq, cnt = np.unique(scan, return_counts=True)
    order = uniq[np.argsort(-cnt)]
    code = {s: i for i, s in enumerate(order)}
    s = torch.tensor([code[x] for x in scan])
    return f, t, s


def run_seed(features, targets, site_ids, seed):
    """One leave-one-site-out pass at a fixed seed.

    Trains PlainMLP per fold, full batch, no weight decay, for max(CKPTS)
    epochs; at each checkpoint epoch records every training subject's CE.
    Checkpoint scores are averaged over the folds in which a subject is in
    training and graded against the held-out binary error.

    Returns:
        Dict mapping checkpoint epoch to {"rho", "auroc", "mean_train_loss"}.
    """
    n_sites = int(site_ids.max()) + 1
    N = len(features)
    ssum = {ep: np.zeros(N) for ep in CKPTS}
    cnt = np.zeros(N)
    pred = np.full(N, -1)
    tested = np.zeros(N, bool)
    mtl = {ep: [] for ep in CKPTS}
    X = features.to(DEVICE)
    y = targets.long().to(DEVICE)
    for held in range(n_sites):
        tr = (site_ids != held).numpy()
        te = (site_ids == held).numpy()
        if te.sum() < 5 or tr.sum() < 20:
            continue
        idx = np.where(tr)[0]
        torch.manual_seed(seed)
        m = PlainMLP(features.shape[1], 2).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=0.0)
        Xt, yt = X[tr], y[tr]
        m.train()
        for ep in range(1, max(CKPTS) + 1):
            opt.zero_grad()
            F.cross_entropy(m(Xt), yt).backward()
            opt.step()
            if ep in CKPTS:
                with torch.no_grad():
                    losses = F.cross_entropy(m(Xt), yt, reduction="none").cpu().numpy()
                ssum[ep][idx] += losses
                mtl[ep].append(float(losses.mean()))
        cnt[idx] += 1
        m.eval()
        with torch.no_grad():
            pred[te] = m(X[te]).argmax(1).cpu().numpy()
        tested[te] = True
    err = (pred != targets.numpy().astype(int)).astype(float)
    out = {}
    for ep in CKPTS:
        sc = (ssum[ep] / np.maximum(cnt, 1.0))[tested]
        e = err[tested]
        rho = float(spearmanr(sc, e)[0])
        auc = float(roc_auc_score(e, sc)) if 0 < e.sum() < len(e) else float("nan")
        out[ep] = {"rho": rho, "auroc": auc, "mean_train_loss": float(np.mean(mtl[ep]))}
    return out


def main():
    """Run both cohorts over SEEDS seeds and write the results JSON."""
    res = {}
    for name, loader in [("adni", load_adni), ("oasis3", load_oasis3_unique)]:
        f, t, s = loader()
        print(f"[{name}] N={len(f)} d={f.shape[1]} E={int(s.max())+1}", flush=True)
        per = {ep: {"rho": [], "auroc": [], "mtl": []} for ep in CKPTS}
        for k in range(SEEDS):
            r = run_seed(f, t, s, SEED_BASE + k)
            for ep in CKPTS:
                per[ep]["rho"].append(r[ep]["rho"])
                per[ep]["auroc"].append(r[ep]["auroc"])
                per[ep]["mtl"].append(r[ep]["mean_train_loss"])
            print(f"  seed {k} done", flush=True)
        res[name] = {str(ep): {
            "rho_mean": float(np.mean(v["rho"])), "rho_std": float(np.std(v["rho"])),
            "auroc_mean": float(np.nanmean(v["auroc"])), "auroc_std": float(np.nanstd(v["auroc"])),
            "mean_train_loss": float(np.mean(v["mtl"]))} for ep, v in per.items()}
        for ep in CKPTS:
            r = res[name][str(ep)]
            print(f"  ep{ep:5d}: train_loss={r['mean_train_loss']:.5f} "
                  f"rho={r['rho_mean']:.3f}±{r['rho_std']:.3f} AUROC={r['auroc_mean']:.3f}",
                  flush=True)
    json.dump(res, open(RES, "w"), indent=2)
    print("wrote", RES)


if __name__ == "__main__":
    main()
