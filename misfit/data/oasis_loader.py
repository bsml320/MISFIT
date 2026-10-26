"""OASIS (Open Access Series of Imaging Studies) dataset loader.

Loads OASIS-1 VBM (Voxel-Based Morphometry) gray matter maps via nilearn's
fetch_oasis_vbm fetcher. Creates multi-environment structure by binning
subjects by age or using cross-validation folds as pseudo-environments.

Target: CDR (Clinical Dementia Rating) binarized as 0 (healthy) vs 1 (demented).
Features: GM density maps projected to ROI atlas (parcellation-based features).

OASIS-1 has ~400 subjects aged 18-96, with ~100 having mild-to-moderate dementia.

Requires: pip install nilearn
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from misfit.data.mri_multisite import MRIData

logger = logging.getLogger("fire_mechanist")


def load_oasis(
    data_dir: str | Path | None = None,
    n_envs: int = 4,
    env_strategy: str = "age_quartile",
    feature_type: str = "vbm_gm",
    n_parcels: int = 100,
    min_subjects_per_env: int = 10,
    normalize: bool = True,
    normalize_mode: str = "per_site",
) -> MRIData:
    """Load OASIS-1 VBM data and return MRIData.

    Args:
        data_dir: Directory to cache downloaded data. None = nilearn default.
        n_envs: Number of environments (used for age_quartile strategy).
        env_strategy: How to create environments:
            - "age_quartile": bin subjects by age quartiles (default)
            - "gender": male vs female as environments
            - "random": random assignment (baseline)
        feature_type: "vbm_gm" for gray matter VBM features.
        n_parcels: Number of parcels for atlas-based feature reduction.
            If 0, use raw VBM (very high-dimensional).
        min_subjects_per_env: Minimum subjects per environment.
        normalize: Whether to z-score features.
        normalize_mode: "per_site" or "global".

    Returns:
        MRIData with OASIS subjects.
    """
    from nilearn.datasets import fetch_oasis_vbm

    logger.info("Fetching OASIS VBM data...")

    oasis = fetch_oasis_vbm(data_dir=data_dir)

    # oasis has: gray_matter_maps, white_matter_maps, ext_vars (DataFrame)
    # ext_vars columns: age, education, ses (socioeconomic status),
    #   mmse, cdr, e_tiv, n_wbv, asf, gender (M/F)
    gm_maps = oasis.gray_matter_maps  # list of file paths
    ext_vars = oasis.ext_vars  # DataFrame

    n_total = len(gm_maps)
    logger.info("OASIS: %d total subjects", n_total)

    # Extract features from VBM maps
    features_np = _extract_vbm_features(gm_maps, n_parcels)

    # Target: CDR (Clinical Dementia Rating)
    # CDR = 0 -> healthy, CDR > 0 -> some dementia
    cdr = ext_vars["cdr"].values.astype(float)
    # Handle NaN CDR
    valid_mask = ~np.isnan(cdr)
    features_np = features_np[valid_mask]
    ext_vars_valid = ext_vars.iloc[valid_mask].reset_index(drop=True)
    cdr = cdr[valid_mask]

    targets_np = (cdr > 0).astype(np.float32)

    logger.info(
        "OASIS after CDR filter: %d subjects, %.1f%% demented",
        len(targets_np), 100.0 * targets_np.mean(),
    )

    # Create environments
    if env_strategy == "age_quartile":
        ages = ext_vars_valid["age"].values.astype(float)
        quantiles = np.quantile(ages, np.linspace(0, 1, n_envs + 1))
        env_ids_np = np.digitize(ages, quantiles[1:-1])
    elif env_strategy == "gender":
        genders = ext_vars_valid["gender"].values
        env_ids_np = np.array([0 if g == "M" else 1 for g in genders])
        n_envs = 2
    elif env_strategy == "random":
        rng = np.random.RandomState(42)
        env_ids_np = rng.randint(0, n_envs, size=len(targets_np))
    else:
        raise ValueError(f"Unknown env_strategy: {env_strategy}")

    # Filter environments with too few subjects
    unique_envs = sorted(set(env_ids_np))
    env_counts = {e: np.sum(env_ids_np == e) for e in unique_envs}
    valid_envs = [e for e, c in env_counts.items() if c >= min_subjects_per_env]

    if len(valid_envs) < 2:
        raise ValueError(
            f"Only {len(valid_envs)} envs have >= {min_subjects_per_env} subjects. "
            f"Counts: {env_counts}"
        )

    keep = np.isin(env_ids_np, valid_envs)
    features_np = features_np[keep]
    targets_np = targets_np[keep]
    env_ids_np = env_ids_np[keep]

    # Re-encode environments to 0..n_envs-1
    env_map = {e: i for i, e in enumerate(sorted(set(env_ids_np)))}
    env_ids_np = np.array([env_map[e] for e in env_ids_np])
    n_envs_final = len(env_map)

    # Build site names from strategy
    if env_strategy == "age_quartile":
        site_names = [f"age_q{i}" for i in range(n_envs_final)]
    elif env_strategy == "gender":
        site_names = ["male", "female"]
    else:
        site_names = [f"env_{i}" for i in range(n_envs_final)]

    n_subjects = len(targets_np)

    # Convert to tensors
    features = torch.tensor(features_np, dtype=torch.float32)
    targets = torch.tensor(targets_np, dtype=torch.float32)
    site_ids = torch.tensor(env_ids_np, dtype=torch.long)
    subject_ids = torch.arange(n_subjects, dtype=torch.long)

    # Normalize
    if normalize:
        if normalize_mode == "per_site":
            for s in range(n_envs_final):
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
    subject_site_matrix = torch.zeros(n_subjects, n_envs_final)
    for i in range(n_subjects):
        subject_site_matrix[i, site_ids[i]] = 1.0

    d = features.shape[1]
    feature_names = [f"parcel_{i}" for i in range(d)]

    logger.info(
        "OASIS loaded: %d subjects, %d envs (%s), %d features, %.1f%% demented",
        n_subjects, n_envs_final, env_strategy, d, 100.0 * targets.mean().item(),
    )
    for s in range(n_envs_final):
        mask = site_ids == s
        logger.info(
            "  %s: %d subjects, %.1f%% demented",
            site_names[s], mask.sum().item(),
            100.0 * targets[mask].mean().item(),
        )

    return MRIData(
        features=features,
        targets=targets,
        subject_ids=subject_ids,
        site_ids=site_ids,
        n_sites=n_envs_final,
        n_subjects=n_subjects,
        feature_names=feature_names,
        site_names=site_names,
        subject_site_matrix=subject_site_matrix,
    )


def _extract_vbm_features(
    gm_maps: list[str],
    n_parcels: int,
) -> np.ndarray:
    """Extract ROI-averaged features from VBM gray matter maps.

    Args:
        gm_maps: List of NIfTI file paths.
        n_parcels: Number of parcels. If 0, use flattened masked voxels.

    Returns:
        (n_subjects, n_features) array.
    """
    from nilearn.maskers import NiftiLabelsMasker, NiftiMasker

    if n_parcels > 0:
        # Use atlas parcellation for dimensionality reduction
        from nilearn.datasets import fetch_atlas_schaefer_2018

        try:
            atlas = fetch_atlas_schaefer_2018(n_rois=n_parcels)
            atlas_file = atlas.maps
        except Exception:
            # Fallback: use Harvard-Oxford if Schaefer fails
            from nilearn.datasets import fetch_atlas_harvard_oxford

            atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
            atlas_file = atlas.maps
            logger.info("Using Harvard-Oxford atlas as fallback")

        masker = NiftiLabelsMasker(labels_img=atlas_file, standardize=False)
        features = masker.fit_transform(gm_maps)
    else:
        # Raw masked voxels (high-dimensional)
        masker = NiftiMasker(standardize=False, memory="nilearn_cache")
        features = masker.fit_transform(gm_maps)

    # Handle any NaN
    features = np.nan_to_num(features, nan=0.0)

    return features.astype(np.float32)
