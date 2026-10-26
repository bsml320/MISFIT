"""CRT study: volume loading and the frozen CNN fit with per-epoch traces.

Recipe (identical for source and target banks, from the documented backbone):
  smwc1 gray-matter volume -> avg_pool3d(2) -> global scalar z-score with
  constants FROZEN per cohort (computed once over the full cohort, recorded in
  the config; the two-scalar leakage is disclosed in the protocol);
  CNN3D, 50 epochs, batch 16, Adam lr 1e-4, coupled weight_decay 1e-4
  (equals the gradient of (lam/2)||theta||^2 with lam = 1e-4), no augmentation,
  no early stopping. Final checkpoint = epoch 50.

Per-epoch traces recorded by full-cohort deterministic eval (GroupNorm: eval
mode == train mode): margin m_i = z_{y_i} - z_{1-y_i} every epoch (AUM = mean
over epochs 1..50, training members only, higher = easier), EL2N at epoch 1
(||p - onehot(y)||_2), CE at checkpoints {10, 25, 50}.

Saves are atomic (tmp ends .npz then rename) and carry the config hash.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from misfit.crt_study.code.model import CNN3D

EPOCHS, BS, LR, WD = 50, 16, 1e-4, 1e-4
CKPTS = (10, 25, 50)
EL2N_EP = 1
DEV = "cuda"


def load_volumes(paths, pool=2):
    def one(p):
        v = nib.load(p).get_fdata(dtype=np.float32)
        t = torch.from_numpy(v)[None, None]
        return F.avg_pool3d(t, pool)[0, 0]
    with ThreadPoolExecutor(max_workers=16) as ex:
        vols = list(ex.map(one, list(paths)))
    return torch.stack(vols)[:, None]                  # N,1,D,H,W


def norm_constants(X):
    return float(X.mean()), float(X.std())


def apply_norm(X, mu, sd):
    return (X - mu) / (sd + 1e-6)


def cfg_dict(cohort, mu, sd, shape):
    arch = {k: list(v.shape) for k, v in CNN3D().state_dict().items()}
    return {"cohort": cohort, "epochs": EPOCHS, "bs": BS, "lr": LR, "wd": WD,
            "ckpts": list(CKPTS), "el2n_ep": EL2N_EP, "pool": 2,
            "norm_mu": round(mu, 6), "norm_sd": round(sd, 6),
            "input_shape": list(shape), "arch": "CNN3D_backbone",
            "arch_shapes": {k: arch[k] for k in sorted(arch)}, "opt": "Adam"}


def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def eval_all(m, X, yt, bs=128):
    """Deterministic full-cohort logits (eval mode; GroupNorm, no dropout)."""
    m.eval()
    out = torch.empty(len(X), 2)
    for i in range(0, len(X), bs):
        out[i:i + bs] = m(X[i:i + bs].to(DEV)).cpu()
    return out


def fit(X, y, train_idx, seed, want_traces=True):
    """One frozen-recipe fit. Returns state dict (cpu), traces, final logits."""
    yt = torch.from_numpy(np.asarray(y)).long()
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    m = CNN3D().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    N = len(y)
    margins = np.zeros((EPOCHS, N), np.float32) if want_traces else None
    el2n = None
    ce_ck = {}
    for ep in range(1, EPOCHS + 1):
        m.train()
        perm = rng.permutation(train_idx)
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            opt.zero_grad()
            F.cross_entropy(m(X[b].to(DEV)), yt[b].to(DEV)).backward()
            opt.step()
        if want_traces or ep in CKPTS:
            z = eval_all(m, X, yt)
            zy = z.gather(1, yt[:, None])[:, 0]
            zo = z.gather(1, 1 - yt[:, None])[:, 0]
            if want_traces:
                margins[ep - 1] = (zy - zo).numpy()
            if ep == EL2N_EP:
                p = torch.softmax(z, 1)
                onehot = F.one_hot(yt, 2).float()
                el2n = (p - onehot).norm(dim=1).numpy().astype(np.float32)
            if ep in CKPTS:
                ce_ck[ep] = F.cross_entropy(z, yt, reduction="none").numpy().astype(np.float32)
    state = {k: v.cpu() for k, v in m.state_dict().items()}
    return state, {"margins": margins, "el2n": el2n, "ce_ck": ce_ck}, eval_all(m, X, yt).numpy()


def save_fit(path, state, traces, logits, train_mask, chash, seed, wall_s):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrs = {f"w_{k}": state[k].numpy() for k in state}
    if traces["margins"] is not None:
        arrs["margins"] = traces["margins"]
        arrs["el2n"] = traces["el2n"]
        for ep, v in traces["ce_ck"].items():
            arrs[f"ce_ep{ep}"] = v
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, logits=logits, train_mask=train_mask,
                        cfg_hash=np.array(chash), seed=np.array(seed),
                        wall_s=np.array(wall_s), **arrs)
    os.replace(tmp, path)


def load_state(path, chash):
    d = np.load(path, allow_pickle=False)
    if str(d["cfg_hash"]) != chash:
        raise RuntimeError(f"{path}: config hash mismatch")
    state = {k[2:]: torch.from_numpy(d[k]) for k in d.files if k.startswith("w_")}
    return d, state
