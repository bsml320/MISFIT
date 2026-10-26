"""UK Biobank sMRI loader for the structural-misfit reliability harness.

The UKBB scans are SPM DARTEL VBM (`smwc1pT1.nii`, descrip 'Smoothed (3x3x3) Dartel normed',
121x145x121 @ 1.5 mm) -- the SAME recipe as the ADNI data -- so this mirrors
`run_adni_smri.load_adni_smri`: GM mask -> flatten -> PCA-50 -> global z-normalization.
Target = sex; environments = age quartiles (as OASIS-1 uses age strata).

ATTRIBUTION: per <DATA_SHARE>/UKBB/README, any use of this data requires Anton Orlichenko
and Dr. Ray Su (Tulane) as co-authors.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

UKBB = Path(_resolve_path("<DATA_SHARE>/UKBB"))
SMRI = UKBB / "smri" / "preprocessed" / "t1_20252_2"
LABELS = UKBB / "phenotype" / "clean_files" / "label.csv"
CACHE = Path(_resolve_path("<REPO_ROOT>/misfit/results/ukbb"))


@dataclass
class UKBBData:
    """Container for the loaded UKBB subset."""

    features: torch.Tensor   # (N, pca_dim) z-normalized PCA scores
    targets: torch.Tensor    # (N,) sex 0/1
    site_ids: torch.Tensor   # (N,) age-quartile environment 0..n_envs-1
    age: np.ndarray          # (N,) age in years
    n_voxels: int


def _smwc1(file_name: str) -> Path:
    """Path to the smoothed modulated GM map for one subject directory."""
    return SMRI / file_name / "t1" / "smwc1pT1.nii"


def _brain_mask(files, threshold=0.1, max_subjects=50, seed=42) -> np.ndarray:
    """Boolean GM mask: mean of up to max_subjects random volumes > threshold."""
    rng = np.random.RandomState(seed)
    pick = rng.choice(len(files), min(max_subjects, len(files)), replace=False)
    mean_vol = None
    for i in pick:
        d = nib.load(files[i]).get_fdata(dtype=np.float32)
        mean_vol = d.astype(np.float64) if mean_vol is None else mean_vol + d
    mean_vol /= len(pick)
    return mean_vol > threshold


def load_ukbb(n_subjects: int = 5000, n_envs: int = 4, pca_dim: int = 50,
              seed: int = 42, jobs: int = 8) -> UKBBData:
    """Load an n_subjects subset of UKBB VBM as PCA features.

    Pipeline: read the label CSV, keep subjects with valid sex/age and an
    existing GM map, draw a seeded random subset, mask and flatten the
    volumes, project to pca_dim components, z-normalize globally, and cache
    the result as an .npz keyed by (n_subjects, n_envs, pca_dim, seed).

    Args:
        n_subjects: Subset size drawn from the eligible cohort.
        n_envs: Number of age-quantile environments.
        pca_dim: Number of PCA components kept.
        seed: Seed for subject sampling and the mask subsample.
        jobs: Thread count for volume loading.

    Returns:
        UKBBData with PCA features, sex targets, and age-quartile site ids.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"ukbb_n{n_subjects}_e{n_envs}_p{pca_dim}_s{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        return UKBBData(torch.tensor(z["features"]), torch.tensor(z["targets"]),
                        torch.tensor(z["site_ids"]), z["age"], int(z["n_voxels"]))

    rows = list(csv.DictReader(open(LABELS)))
    recs = [(str(_smwc1(r["file_name"])), int(r["sex"]), int(r["age"]))
            for r in rows
            if r["sex"] in ("0", "1") and r["age"].isdigit() and _smwc1(r["file_name"]).exists()]
    rng = np.random.RandomState(seed)
    recs = [recs[i] for i in rng.permutation(len(recs))[:n_subjects]]
    files = [r[0] for r in recs]
    sex = np.array([r[1] for r in recs], dtype=np.int64)
    age = np.array([r[2] for r in recs], dtype=np.int64)

    # age-quartile environments
    edges = np.quantile(age, np.linspace(0, 1, n_envs + 1)[1:-1])
    env = np.digitize(age, edges).astype(np.int64)

    mask = _brain_mask(files, seed=seed)
    n_voxels = int(mask.sum())

    def _load(f):
        return nib.load(f).get_fdata(dtype=np.float32)[mask]
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        feats = np.stack(list(ex.map(_load, files)))     # (N, n_voxels) float32

    # PCA-50 then global z-norm -- as ADNI. Truncated randomized SVD (top pca_dim only) in
    # float32 keeps this O(m*n*k) and memory-feasible at the full 42k cohort (~75 GB vs ~150 GB).
    X = torch.from_numpy(feats)                       # float32 (N, n_voxels)
    mean = X.mean(0)
    U, S, _ = torch.svd_lowrank(X - mean, q=pca_dim + 8, niter=6)
    pca = U[:, :pca_dim] * S[:pca_dim]
    gmean, gstd = pca.mean(0), pca.std(0).clamp(min=1e-8)
    pca = (pca - gmean) / gstd
    del feats, X

    np.savez(cache, features=pca.numpy(), targets=sex, site_ids=env, age=age, n_voxels=n_voxels)
    return UKBBData(pca, torch.tensor(sex, dtype=torch.float32),
                    torch.tensor(env, dtype=torch.long), age, n_voxels)
