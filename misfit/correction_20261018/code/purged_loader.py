"""Loader for the correction reruns.

Replicates misfit/data/adni_loader.py::load_adni_smri
EXACTLY (same mask RNG, same eligibility, same PCA and global z-norm) while
additionally returning the raw masked voxel matrix (needed for the V2 per-fold
PCA variant) and the participant ID (PTID = subject directory name, validated
against ^\\d{3}_S_\\d{4}$ by the membership audit).

Binding guarantee: the returned `features` are asserted allclose (atol 1e-5)
to a direct load_adni_smri() call the first time the cache is built, and the
cache stores the historical features so later runs use the identical matrix.

Cache (participant-level => DUA local-only, never packaged):
  correction_20261018/reruns/adni_cache_local.npz
"""

from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))

COR = Path(_resolve_path("<REPO_ROOT>/misfit/correction_20261018"))
CACHE = COR / "reruns/adni_cache_local.npz"


def load() -> dict:
    """Return the cohort dict (raw voxels, historical features, targets,
    site_ids, ptids, site_names); build and binding-verify the cache on
    first call."""
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=False)
        return {
            "raw": d["raw"], "features": torch.tensor(d["features"]),
            "targets": torch.tensor(d["targets"]),
            "site_ids": torch.tensor(d["site_ids"]),
            "ptids": d["ptids"].astype(str), "site_names": list(d["site_names"].astype(str)),
        }

    from misfit.data.adni_loader import (
        IMPAIRED_GROUPS, NII_DIR, PROTOCOLS, _create_brain_mask, load_adni_smri,
    )
    import nibabel as nib
    import pandas as pd
    from misfit.data.adni_loader import LABELS_CSV

    labels_df = pd.read_csv(LABELS_CSV)
    label_lookup: dict[tuple[str, str], str] = {}
    for _, row in labels_df.iterrows():
        key = (str(row["dataset_id"]).strip(), str(row["file_name"]).strip())
        if key not in label_lookup:
            label_lookup[key] = str(row["Group"]).strip()

    mask = _create_brain_mask(NII_DIR, PROTOCOLS)
    raw_list, lab_list, site_list, ptid_list, site_names = [], [], [], [], []
    site_idx = 0
    for protocol in PROTOCOLS:
        proto_dir = NII_DIR / protocol
        if not proto_dir.exists():
            continue
        feats, labs, pts = [], [], []
        for subj_dir in sorted(proto_dir.iterdir()):
            smwc1 = subj_dir / "smwc1pT1.nii"
            if not smwc1.exists():
                continue
            key = (protocol, subj_dir.name)
            if key not in label_lookup:
                continue
            group = label_lookup[key]
            if group not in {"CN"} | IMPAIRED_GROUPS:
                continue
            feats.append(nib.load(smwc1).get_fdata(dtype=np.float32)[mask])
            labs.append(0 if group == "CN" else 1)
            pts.append(subj_dir.name)
        if len(feats) < 30:
            continue
        raw_list.append(np.array(feats))
        lab_list.append(np.array(labs))
        site_list.append(np.full(len(feats), site_idx, dtype=np.int64))
        ptid_list.extend(pts)
        site_names.append(protocol)
        site_idx += 1

    raw = np.concatenate(raw_list, 0)
    labels = np.concatenate(lab_list, 0)
    sites = np.concatenate(site_list, 0)

    # historical global PCA + z-norm, identical ops
    f64 = torch.tensor(raw, dtype=torch.float64)
    f64 = f64 - f64.mean(dim=0)
    U, S, _ = torch.linalg.svd(f64, full_matrices=False)
    pca = (U[:, :50] * S[:50]).float()
    gm, gs = pca.mean(dim=0), pca.std(dim=0).clamp(min=1e-8)
    pca = (pca - gm) / gs

    # binding check against the untouched historical loader (engineering
    # review m6: bitwise equality verified empirically; the cache stores the
    # HISTORICAL loader's tensor)
    mri = load_adni_smri()
    assert mri.features.shape == pca.shape, "shape mismatch vs historical loader"
    assert torch.equal(mri.features, pca), "feature mismatch vs historical loader"
    assert torch.equal(mri.site_ids, torch.tensor(sites)), "site mismatch"
    assert torch.equal(mri.targets.long(), torch.tensor(labels).long()), "label mismatch"

    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        CACHE, raw=raw, features=mri.features.numpy(),
        targets=labels.astype(np.float32),
        site_ids=sites, ptids=np.array(ptid_list), site_names=np.array(site_names),
    )
    print(f"cache built and binding-verified: {CACHE}")
    return load()


def fold_pca(raw: np.ndarray, train_idx: np.ndarray, pca_dim: int = 50):
    """V2 per-fold PCA + z-norm fit on the purged training pool only.

    Same construction as the historical global one restricted to training:
    center by train mean, right-singular basis of the centered train matrix,
    z-norm by train statistics; held-out rows are projected.
    Returns a function mapping row indices -> float32 feature tensor.
    """
    tr = torch.tensor(raw[train_idx], dtype=torch.float64)
    mean = tr.mean(dim=0)
    trc = tr - mean
    _, _, Vh = torch.linalg.svd(trc, full_matrices=False)
    V = Vh[:pca_dim].T  # (voxels, pca_dim)
    tr_p = (trc @ V).float()
    gm, gs = tr_p.mean(dim=0), tr_p.std(dim=0).clamp(min=1e-8)

    def project(idx: np.ndarray) -> torch.Tensor:
        x = torch.tensor(raw[idx], dtype=torch.float64) - mean
        return (((x @ V).float()) - gm) / gs

    return project
