"""Shared pieces for the deferred-rerun scripts (plan/RERUN_PLAN.md C1-C3).

Cohort: the binding-verified correction cache (731 acquisitions / 547
participants). Folds: purge_masks from the verified correction harness.
Per-fold features: purged_loader.fold_pca (source-only center + PCA-50 +
z-norm, float64 SVD on CPU), computed once per fold and shared across seeds
(the projection is seed-independent).
"""

from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import sys

import numpy as np

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
sys.path.insert(0, _resolve_path("<REPO_ROOT>/misfit/correction_20261018/code"))

from purged_harness import purge_masks  # noqa: E402
from purged_loader import fold_pca, load  # noqa: E402

__all__ = ["load", "purge_masks", "fold_pca", "purged_fold_projections"]


def purged_fold_projections(raw: np.ndarray, site_ids, ptids: np.ndarray,
                            pca_dim: int = 50) -> dict:
    """For each eligible held-out protocol: purged train mask, test mask, and
    the full-cohort projection into that fold's source-only feature space."""
    n_sites = int(site_ids.max()) + 1
    n = len(ptids)
    folds = {}
    for held in range(n_sites):
        train_mask, test_mask = purge_masks(site_ids, ptids, held)
        if test_mask.sum() < 5 or train_mask.sum() < 20:
            continue
        proj = fold_pca(raw, np.where(train_mask.numpy())[0], pca_dim)
        folds[held] = {"train": train_mask.numpy(), "test": test_mask.numpy(),
                       "Z": proj(np.arange(n))}
        print(f"  fold {held}: purged train={int(train_mask.sum())} "
              f"test={int(test_mask.sum())}, source-only PCA ready", flush=True)
    return folds
