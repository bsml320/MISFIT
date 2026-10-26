"""3D-CNN backbone test (reviewer preempt): does the per-subject training-loss reliability
signal survive when the classifier is an image-level 3D CNN rather than the PCA-MLP?

Light 3D CNN (4 conv blocks, 8-64 ch, GAP head) on the leakage-free OASIS-3 cohort
(N=1360, one earliest T1w session per participant), operating directly on the smwc1
gray-matter volumes (avg-pooled 2x -> 60x72x60). Leave-one-scanner-out, 3 seeds.
Score = the CNN's own per-subject training cross-entropy, recorded at epoch checkpoints
and aggregated across the folds in which the subject is in training; graded against the
held-out scanner errors. GPU (set via CUDA_VISIBLE_DEVICES, default 3).
Norm: GroupNorm (batch-independent; BatchNorm at batch 16 in 3D corrupts eval-mode per-subject CE).
Output: results/oasis3/oasis3_cnn_backbone.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

# CUDA visibility is selected by the caller.
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

WORK = _resolve_path("<REPO_ROOT>/data/oasis3/route_b_work")
RES = _resolve_path("<REPO_ROOT>/misfit/results/oasis3/oasis3_cnn_backbone.json")
DEV = "cuda"
SEEDS, SEED_BASE = 3, 42
EPOCHS, BS = 50, 16
CKPTS = [10, 25, 50]


def load_cohort():
    """Leakage-free OASIS-3 cohort: one earliest session per participant.

    Returns:
        (session ids, labels, site codes, scanner names ordered by frequency).
    """
    d = np.load(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz"), allow_pickle=True)
    subj = d["subjects"].astype(str)
    sess = d["sessions"].astype(str)
    day = np.array([int(re.search(r"_d(\d+)$", s).group(1)) for s in sess])
    keep = np.array(sorted(np.where(subj == u)[0][np.argmin(day[subj == u])]
                           for u in np.unique(subj)))
    sess = sess[keep]
    lab = d["label"].astype(int)[keep]
    scan = d["scanner"].astype(str)[keep]
    uniq, cnt = np.unique(scan, return_counts=True)
    order = uniq[np.argsort(-cnt)]
    code = {s: i for i, s in enumerate(order)}
    site = np.array([code[x] for x in scan])
    return sess, lab, site, [str(o) for o in order]


def load_volumes(sess):
    """Load smwc1 gray-matter volumes for the given sessions.

    Each volume is 2x average-pooled (121x145x121 -> 60x72x60); the stack is
    standardized with a single global mean/std over all voxels and subjects.

    Returns:
        Tensor of shape (N, 1, 60, 72, 60).
    """
    def one(s):
        v = nib.load(f"{WORK}/{s}/smwc1pT1.nii").get_fdata(dtype=np.float32)
        t = torch.from_numpy(v)[None, None]
        return F.avg_pool3d(t, 2)[0, 0]          # 121x145x121 -> 60x72x60
    with ThreadPoolExecutor(max_workers=16) as ex:
        vols = list(ex.map(one, sess))
    X = torch.stack(vols)
    X = (X - X.mean()) / (X.std() + 1e-6)
    return X[:, None]                             # N,1,60,72,60


class CNN3D(nn.Module):
    """Light 3D CNN: four Conv3d/GroupNorm/ReLU/MaxPool3d blocks (8-64
    channels), global average pooling, linear 2-class head."""

    def __init__(self):
        super().__init__()
        ch = [1, 8, 16, 32, 64]
        layers = []
        for i in range(4):
            layers += [nn.Conv3d(ch[i], ch[i + 1], 3, padding=1),
                       nn.GroupNorm(min(8, ch[i + 1]), ch[i + 1]), nn.ReLU(), nn.MaxPool3d(2)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        return self.head(self.body(x).mean(dim=(2, 3, 4)))


def batched_ce(m, X, yt, idx):
    """Per-example CE for rows idx, evaluated in device batches of 64."""
    out = np.zeros(len(idx))
    with torch.no_grad():
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            l = F.cross_entropy(m(X[b].to(DEV)), yt[b].to(DEV), reduction="none")
            out[i:i + len(b)] = l.cpu().numpy()
    return out


def run_seed(X, y, site, seed):
    """One leave-one-scanner-out pass at a fixed seed.

    Per fold: trains CNN3D for EPOCHS epochs; at each checkpoint epoch records
    per-subject training CE in eval mode (GroupNorm makes this batch-size
    independent) and predicts the held-out scanner. Checkpoint scores are
    fold-averaged and graded against held-out binary error, pooled and within
    scanner.

    Returns:
        Dict mapping checkpoint epoch to metrics, plus "base_acc".
    """
    n_sites = int(site.max()) + 1
    N = len(y)
    ssum = {ep: np.zeros(N) for ep in CKPTS}
    cnt = np.zeros(N)
    pred = np.full(N, -1)
    tested = np.zeros(N, bool)
    mtl = {ep: [] for ep in CKPTS}
    yt = torch.from_numpy(y).long()
    for held in range(n_sites):
        tr = np.where(site != held)[0]
        te = np.where(site == held)[0]
        if len(te) < 5:
            continue
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed * 100 + held)
        m = CNN3D().to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
        for ep in range(1, EPOCHS + 1):
            m.train()
            perm = rng.permutation(tr)
            for i in range(0, len(perm), BS):
                b = perm[i:i + BS]
                opt.zero_grad()
                F.cross_entropy(m(X[b].to(DEV)), yt[b].to(DEV)).backward()
                opt.step()
            if ep in CKPTS:
                m.eval()
                losses = batched_ce(m, X, yt, tr)
                ssum[ep][tr] += losses
                mtl[ep].append(float(losses.mean()))
        cnt[tr] += 1
        m.eval()
        with torch.no_grad():
            for i in range(0, len(te), 64):
                b = te[i:i + 64]
                pred[b] = m(X[b].to(DEV)).argmax(1).cpu().numpy()
        tested[te] = True
        print(f"    held scanner {held}: done (test n={len(te)})", flush=True)
    err = (pred != y).astype(float)
    out = {}
    for ep in CKPTS:
        sc = (ssum[ep] / np.maximum(cnt, 1.0))
        m_ = tested
        rho = float(spearmanr(sc[m_], err[m_])[0])
        auc = float(roc_auc_score(err[m_], sc[m_])) if 0 < err[m_].sum() < m_.sum() else float("nan")
        # within-scanner rho at this checkpoint
        wr = []
        for e in np.unique(site[m_]):
            mm = m_ & (site == e)
            ee = err[mm]
            if len(ee) >= 8 and 0 < ee.sum() < len(ee):
                r = spearmanr(sc[mm], ee)[0]
                if not np.isnan(r):
                    wr.append(r)
        out[ep] = {"rho": rho, "auroc": auc, "mean_train_loss": float(np.mean(mtl[ep])),
                   "within_scanner_rho": float(np.mean(wr)), "n_scored": len(wr),
                   "frac_positive": float(np.mean([r > 0 for r in wr]))}
    out["base_acc"] = float(1 - err[tested].mean())
    return out


def main():
    """Load cohort and volumes, run SEEDS seeds, aggregate, write the JSON."""
    sess, y, site, order = load_cohort()
    print(f"OASIS-3 leakage-free: N={len(y)} impaired={y.sum()} scanners={order}", flush=True)
    print("loading volumes ...", flush=True)
    X = load_volumes(sess)
    print(f"volumes: {tuple(X.shape)} ({X.element_size()*X.nelement()/1e9:.1f} GB)", flush=True)
    per = {ep: {"rho": [], "auroc": [], "ws": [], "fp": [], "mtl": []} for ep in CKPTS}
    base = []
    for k in range(SEEDS):
        print(f"  seed {k}:", flush=True)
        r = run_seed(X, y, site, SEED_BASE + k)
        base.append(r["base_acc"])
        for ep in CKPTS:
            per[ep]["rho"].append(r[ep]["rho"])
            per[ep]["auroc"].append(r[ep]["auroc"])
            per[ep]["ws"].append(r[ep]["within_scanner_rho"])
            per[ep]["fp"].append(r[ep]["frac_positive"])
            per[ep]["mtl"].append(r[ep]["mean_train_loss"])
    res = {"n": int(len(y)), "scanners": order, "epochs": EPOCHS,
           "base_acc_mean": float(np.mean(base))}
    for ep in CKPTS:
        v = per[ep]
        res[f"ep{ep}"] = {
            "pooled_rho_mean": float(np.mean(v["rho"])), "pooled_rho_std": float(np.std(v["rho"])),
            "auroc_mean": float(np.nanmean(v["auroc"])), "auroc_std": float(np.nanstd(v["auroc"])),
            "within_scanner_rho_mean": float(np.mean(v["ws"])),
            "within_scanner_rho_std": float(np.std(v["ws"])),
            "frac_positive": float(np.mean(v["fp"])),
            "mean_train_loss": float(np.mean(v["mtl"]))}
        r = res[f"ep{ep}"]
        print(f"  ep{ep}: train_loss={r['mean_train_loss']:.4f} AUROC={r['auroc_mean']:.3f} "
              f"pooled_rho={r['pooled_rho_mean']:.3f} within-scanner={r['within_scanner_rho_mean']:.3f} "
              f"({r['frac_positive']*100:.0f}% +)", flush=True)
    json.dump(res, open(RES, "w"), indent=2)
    print("wrote", RES)


if __name__ == "__main__":
    main()
