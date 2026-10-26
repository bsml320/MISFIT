"""ADNI gray-matter loader for the paper's corrected and historical controls.

The original loading functions are retained; the unrelated standalone benchmark
runner is excluded. Current manuscript fits use participant-purged folds.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from misfit.data.mri_multisite import MRIData

logger = logging.getLogger("misfit")

PCA_DIM = 50

MIN_SUBJECTS_PER_SITE = 30

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(
    os.environ.get("ADNI_SMRI_DATA_ROOT", str(_PROJECT_ROOT / "data" / "adni" / "smri"))
)

NII_DIR = DATA_ROOT / "nii"

LABELS_CSV = DATA_ROOT / "labels" / "combined.csv"

PROTOCOLS = [
    "ADNI2-GE-initial",
    "ADNI2-Phillips-initial",
    "ADNI2-Siemens-initial",
    "ADNI3-GE-initial",
    "ADNI3-Phillips-initial",
    "ADNI3-Siemens-initial",
    "ADNIGO-GE-initial",
    "ADNIGO-Phillips-initial",
    "ADNIGO-Siemens-initial",
]

IMPAIRED_GROUPS = {"AD", "MCI", "EMCI", "LMCI"}

def _create_brain_mask(
    nii_dir: Path,
    protocols: list[str],
    threshold: float = 0.1,
    max_subjects: int = 50,
) -> np.ndarray:
    """Create a brain mask from a sample of subjects.

    Loads a random subset of gray matter volumes and creates a mask where
    the mean intensity exceeds the threshold.

    Returns:
        Boolean mask array of shape (121, 145, 121).
    """
    all_files: list[Path] = []
    for protocol in protocols:
        proto_dir = nii_dir / protocol
        if not proto_dir.exists():
            continue
        for subj_dir in proto_dir.iterdir():
            smwc1 = subj_dir / "smwc1pT1.nii"
            if smwc1.exists():
                all_files.append(smwc1)

    # Sample subset for mask creation
    rng = np.random.RandomState(42)
    if len(all_files) > max_subjects:
        indices = rng.choice(len(all_files), max_subjects, replace=False)
        sample_files = [all_files[i] for i in indices]
    else:
        sample_files = all_files

    logger.info("Creating brain mask from %d subjects", len(sample_files))

    # Accumulate mean
    mean_vol = None
    for fpath in sample_files:
        data = nib.load(fpath).get_fdata(dtype=np.float32)
        if mean_vol is None:
            mean_vol = np.zeros_like(data, dtype=np.float64)
        mean_vol += data

    mean_vol /= len(sample_files)
    mask = mean_vol > threshold
    n_voxels = int(mask.sum())
    logger.info("Brain mask: %d voxels (%.1f%% of volume)", n_voxels,
                100.0 * n_voxels / mask.size)
    return mask

def load_adni_smri(
    min_subjects: int = MIN_SUBJECTS_PER_SITE,
    pca_dim: int = PCA_DIM,
) -> MRIData:
    """Load ADNI sMRI data across all protocols, PCA-reduce, return MRIData.

    Uses global z-normalization (not per-site) based on fMRI findings that
    per-site normalization removes the site effects FIRE needs.

    Args:
        min_subjects: Minimum subjects per protocol to include as a site.
        pca_dim: Number of PCA components for dimensionality reduction.

    Returns:
        MRIData with PCA-reduced features and protocol-based site assignments.
    """
    # Load labels
    labels_df = pd.read_csv(LABELS_CSV)
    # Build lookup: (dataset_id, file_name) -> Group
    label_lookup: dict[tuple[str, str], str] = {}
    for _, row in labels_df.iterrows():
        key = (str(row["dataset_id"]).strip(), str(row["file_name"]).strip())
        group = str(row["Group"]).strip()
        if key not in label_lookup:
            label_lookup[key] = group

    # Create brain mask
    mask = _create_brain_mask(NII_DIR, PROTOCOLS)
    n_voxels = int(mask.sum())

    all_features = []
    all_labels = []
    all_site_ids = []
    all_subject_ids = []
    site_names = []
    global_subject_map: dict[str, int] = {}
    site_idx = 0

    for protocol in PROTOCOLS:
        proto_dir = NII_DIR / protocol
        if not proto_dir.exists():
            logger.warning("Protocol directory not found: %s", proto_dir)
            continue

        features_list = []
        labels_list = []
        subjects_list = []

        for subj_dir in sorted(proto_dir.iterdir()):
            subj_id = subj_dir.name
            smwc1 = subj_dir / "smwc1pT1.nii"
            if not smwc1.exists():
                continue

            # Look up label
            key = (protocol, subj_id)
            if key not in label_lookup:
                continue
            group = label_lookup[key]
            if group not in {"CN"} | IMPAIRED_GROUPS:
                continue

            # Load volume and apply mask
            data = nib.load(smwc1).get_fdata(dtype=np.float32)
            features_list.append(data[mask])
            labels_list.append(0 if group == "CN" else 1)
            subjects_list.append(subj_id)

        n = len(features_list)
        if n < min_subjects:
            logger.info(
                "Skipping %s: only %d subjects (need %d)",
                protocol, n, min_subjects,
            )
            continue

        feats = np.array(features_list)  # (n, n_voxels)
        labs = np.array(labels_list)
        logger.info(
            "Protocol %s: %d subjects (%d CN, %d impaired)",
            protocol, n, int((labs == 0).sum()), int((labs == 1).sum()),
        )

        for subj in subjects_list:
            if subj not in global_subject_map:
                global_subject_map[subj] = len(global_subject_map)
            all_subject_ids.append(global_subject_map[subj])

        all_features.append(feats)
        all_labels.append(labs)
        all_site_ids.append(np.full(n, site_idx, dtype=np.int64))
        site_names.append(protocol)
        site_idx += 1

    if not all_features:
        raise RuntimeError(
            f"No protocols with >= {min_subjects} subjects. "
            "Check data paths and label matching."
        )

    features_np = np.concatenate(all_features, axis=0)  # (N_total, n_voxels)
    labels_np = np.concatenate(all_labels, axis=0)
    site_ids_np = np.concatenate(all_site_ids, axis=0)
    subject_ids_np = np.array(all_subject_ids, dtype=np.int64)

    n_total = features_np.shape[0]
    n_sites = len(site_names)
    n_subjects = len(global_subject_map)

    logger.info(
        "Total: %d subjects, %d unique, %d sites",
        n_total, n_subjects, n_sites,
    )
    logger.info(
        "Overall: %d CN (%.1f%%), %d impaired (%.1f%%)",
        int((labels_np == 0).sum()),
        100.0 * (labels_np == 0).mean(),
        int((labels_np == 1).sum()),
        100.0 * (labels_np == 1).mean(),
    )

    # PCA in float64 for numerical stability
    logger.info("Running PCA: %d voxels -> %d components", n_voxels, pca_dim)
    features_f64 = torch.tensor(features_np, dtype=torch.float64)
    mean = features_f64.mean(dim=0)
    features_centered = features_f64 - mean
    # SVD-based PCA (more stable than covariance eigendecomposition)
    U, S, _ = torch.linalg.svd(features_centered, full_matrices=False)
    features_pca = (U[:, :pca_dim] * S[:pca_dim]).float()  # (N, pca_dim)

    # Global z-normalization (NOT per-site; preserves site effects for FIRE)
    global_mean = features_pca.mean(dim=0)
    global_std = features_pca.std(dim=0).clamp(min=1e-8)
    features_pca = (features_pca - global_mean) / global_std
    logger.info("Using global z-normalization")

    # Build subject-site matrix
    site_ids_t = torch.tensor(site_ids_np, dtype=torch.long)
    subject_site_matrix = torch.zeros(n_subjects, n_sites)
    for i in range(n_total):
        subject_site_matrix[subject_ids_np[i], site_ids_np[i]] = 1.0

    feature_names = [f"PC_{i}" for i in range(pca_dim)]

    return MRIData(
        features=features_pca,
        targets=torch.tensor(labels_np, dtype=torch.float32),
        subject_ids=torch.tensor(subject_ids_np, dtype=torch.long),
        site_ids=site_ids_t,
        n_sites=n_sites,
        n_subjects=n_subjects,
        feature_names=feature_names,
        site_names=site_names,
        subject_site_matrix=subject_site_matrix,
    )
