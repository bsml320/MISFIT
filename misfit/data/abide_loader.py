"""ABIDE (Autism Brain Imaging Data Exchange) dataset loader.

Fetches ABIDE-I preprocessed data via nilearn, extracts ROI-based features
(mean timeseries or correlation matrix), and returns MRIData for the FIRE pipeline.

Requires: pip install nilearn
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from misfit.data.mri_multisite import MRIData

logger = logging.getLogger("fire_mechanist")

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "abide"


def load_abide(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    feature_type: str = "mean_timeseries",
    atlas: str = "cc200",
    pipeline: str = "cpac",
    quality_checked: bool = True,
    min_subjects_per_site: int = 10,
    normalize: bool = True,
    normalize_mode: str = "per_site",
) -> MRIData:
    """Load ABIDE dataset and return MRIData.

    Args:
        data_dir: Directory to cache downloaded data.
        feature_type: One of "mean_timeseries" (200-d mean ROI signal) or
            "correlation" (flattened upper-triangle of ROI correlation matrix).
        atlas: ROI atlas. "cc200" (200 ROIs) or "cc400" (400 ROIs).
        pipeline: Preprocessing pipeline. One of "cpac", "ccs", "dparsf", "niak".
        quality_checked: If True, only use subjects that passed QC.
        min_subjects_per_site: Drop sites with fewer subjects than this.
        normalize: Whether to z-score features.
        normalize_mode: "per_site" or "global".

    Returns:
        MRIData with ABIDE subjects.
    """
    from nilearn.datasets import fetch_abide_pcp

    data_dir = Path(data_dir)
    derivative = f"rois_{atlas}"

    logger.info(
        "Fetching ABIDE data: pipeline=%s, atlas=%s, feature_type=%s",
        pipeline, atlas, feature_type,
    )

    abide = fetch_abide_pcp(
        data_dir=str(data_dir),
        derivatives=[derivative],
        pipeline=pipeline,
        quality_checked=quality_checked,
        verbose=0,
    )

    pheno = abide.phenotypic
    rois_list = getattr(abide, derivative)
    n_total = len(rois_list)

    logger.info("Loaded %d subjects from ABIDE", n_total)

    # Extract features from ROI timeseries
    features_list = []
    valid_indices = []

    for i in range(n_total):
        roi_ts = rois_list[i]  # (timepoints, n_rois) ndarray
        if roi_ts is None or len(roi_ts) == 0:
            continue

        feat = _extract_features(roi_ts, feature_type)
        if feat is not None and np.isfinite(feat).all():
            features_list.append(feat)
            valid_indices.append(i)

    logger.info("Valid subjects after feature extraction: %d / %d", len(valid_indices), n_total)

    # Extract phenotypic info for valid subjects
    pheno_valid = pheno.iloc[valid_indices].reset_index(drop=True)

    # DX_GROUP: 1 = autism, 2 = control -> remap to 0 = control, 1 = autism
    targets_np = (pheno_valid["DX_GROUP"].values == 1).astype(np.float32)
    site_names_raw = pheno_valid["SITE_ID"].values
    subject_ids_raw = pheno_valid["SUB_ID"].values

    features_np = np.stack(features_list, axis=0).astype(np.float32)

    # Filter sites with too few subjects
    unique_sites = sorted(set(site_names_raw))
    site_counts = {s: np.sum(site_names_raw == s) for s in unique_sites}
    valid_sites = {s for s, c in site_counts.items() if c >= min_subjects_per_site}

    keep_mask = np.array([s in valid_sites for s in site_names_raw])
    features_np = features_np[keep_mask]
    targets_np = targets_np[keep_mask]
    site_names_raw = site_names_raw[keep_mask]
    subject_ids_raw = subject_ids_raw[keep_mask]

    logger.info(
        "After site filter (min=%d): %d subjects, %d sites",
        min_subjects_per_site, len(features_np), len(valid_sites),
    )

    # Encode sites and subjects
    site_names_sorted = sorted(valid_sites)
    site_encoder = {s: i for i, s in enumerate(site_names_sorted)}
    n_sites = len(site_names_sorted)

    subject_ids_unique = sorted(set(subject_ids_raw))
    subject_encoder = {s: i for i, s in enumerate(subject_ids_unique)}
    n_subjects = len(subject_ids_unique)

    # Convert to tensors
    features = torch.tensor(features_np, dtype=torch.float32)
    targets = torch.tensor(targets_np, dtype=torch.float32)
    site_ids = torch.tensor([site_encoder[s] for s in site_names_raw], dtype=torch.long)
    subject_ids = torch.tensor([subject_encoder[s] for s in subject_ids_raw], dtype=torch.long)

    # Normalize
    if normalize:
        if normalize_mode == "per_site":
            for s in range(n_sites):
                mask = site_ids == s
                if mask.sum() > 1:
                    feats = features[mask]
                    mean = feats.mean(dim=0)
                    std = feats.std(dim=0).clamp(min=1e-8)
                    features[mask] = (feats - mean) / std
        else:
            mean = features.mean(dim=0)
            std = features.std(dim=0).clamp(min=1e-8)
            features = (features - mean) / std

    # Build subject-site matrix
    subject_site_matrix = torch.zeros(n_subjects, n_sites)
    for i in range(len(subject_ids)):
        subject_site_matrix[subject_ids[i], site_ids[i]] = 1.0

    d = features.shape[1]
    feature_names = [f"roi_{i}" for i in range(d)]

    logger.info(
        "ABIDE loaded: %d subjects, %d sites, %d features, %.1f%% autism",
        n_subjects, n_sites, d, 100.0 * targets.mean().item(),
    )

    return MRIData(
        features=features,
        targets=targets,
        subject_ids=subject_ids,
        site_ids=site_ids,
        n_sites=n_sites,
        n_subjects=n_subjects,
        feature_names=feature_names,
        site_names=site_names_sorted,
        subject_site_matrix=subject_site_matrix,
    )


def _extract_features(roi_ts: np.ndarray, feature_type: str) -> np.ndarray | None:
    """Extract features from ROI timeseries.

    Args:
        roi_ts: (timepoints, n_rois) array.
        feature_type: "mean_timeseries" or "correlation".

    Returns:
        1-D feature vector or None if extraction fails.
    """
    if roi_ts.ndim != 2 or roi_ts.shape[0] < 10:
        return None

    if feature_type == "mean_timeseries":
        # Mean signal per ROI across time
        return roi_ts.mean(axis=0)

    elif feature_type == "correlation":
        # Pearson correlation matrix -> flattened upper triangle
        n_rois = roi_ts.shape[1]
        # Standardize each ROI timeseries
        std = roi_ts.std(axis=0)
        bad_rois = std < 1e-10
        if bad_rois.sum() > n_rois * 0.1:
            return None  # Too many constant ROIs
        # Replace near-constant ROIs with tiny noise so corrcoef stays finite.
        roi_ts_clean = roi_ts.copy()
        roi_ts_clean[:, bad_rois] = np.random.randn(roi_ts.shape[0], bad_rois.sum()) * 1e-6
        corr = np.corrcoef(roi_ts_clean.T)  # (n_rois, n_rois)
        # Upper triangle (excluding diagonal)
        idx = np.triu_indices(n_rois, k=1)
        return corr[idx]

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}. Use 'mean_timeseries' or 'correlation'.")
