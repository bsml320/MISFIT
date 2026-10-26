"""Multi-site MRI data adapter.

Loads multi-site MRI data into formats compatible with the FIRE pipeline.
Supports CSV (ROI-based features), with pseudobulk approach for multi-env format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class MRIDatasetConfig:
    """Configuration for multi-site MRI dataset."""

    data_dir: str | Path = "."
    format: str = "roi_csv"
    target_column: str = "diagnosis"
    subject_column: str = "subject_id"
    site_column: str = "site"
    feature_columns: list[str] | None = None
    normalize: bool = True
    normalize_mode: str = "per_site"  # "per_site" or "global"
    min_subjects_per_site: int = 10


@dataclass
class MRIData:
    """Container for multi-site MRI data."""

    features: torch.Tensor  # (N_total, d)
    targets: torch.Tensor  # (N_total,)
    subject_ids: torch.Tensor  # (N_total,)
    site_ids: torch.Tensor  # (N_total,)
    n_sites: int
    n_subjects: int
    feature_names: list[str]
    site_names: list[str]
    subject_site_matrix: torch.Tensor  # (n_subjects, n_sites) binary


class MultiSiteMRIDataset(Dataset):
    """PyTorch Dataset over MRIData.

    Yields (features, target, site_id, subject_id) tuples per sample.
    """

    def __init__(self, mri_data: MRIData) -> None:
        self.features = mri_data.features
        self.targets = mri_data.targets
        self.site_ids = mri_data.site_ids
        self.subject_ids = mri_data.subject_ids

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        return (
            self.features[idx],
            self.targets[idx],
            int(self.site_ids[idx].item()),
            int(self.subject_ids[idx].item()),
        )


def load_mri_data(config: MRIDatasetConfig) -> MRIData:
    """Load and preprocess multi-site MRI data from CSV.

    Args:
        config: Dataset configuration.

    Returns:
        MRIData container.
    """
    import pandas as pd

    data_dir = Path(config.data_dir)

    if config.format == "roi_csv":
        # Find CSV files
        csv_files = list(data_dir.glob("*.csv"))
        if len(csv_files) == 1:
            df = pd.read_csv(csv_files[0])
        elif (data_dir / "data.csv").exists():
            df = pd.read_csv(data_dir / "data.csv")
        else:
            raise FileNotFoundError(f"No CSV file found in {data_dir}")
    else:
        raise ValueError(
            f"Unsupported format: {config.format}. Currently only 'roi_csv' is supported."
        )

    return _process_dataframe(df, config)


def _process_dataframe(df, config: MRIDatasetConfig) -> MRIData:
    """Process a pandas DataFrame into MRIData.

    Steps: drop sites below the min-subject threshold, encode site and
    subject ids as contiguous integers, select feature columns, optionally
    z-score (per site or globally), and build the subject-site matrix.
    """

    # Encode sites
    site_encoder = {s: i for i, s in enumerate(sorted(df[config.site_column].unique()))}
    site_names = sorted(df[config.site_column].unique())

    # Filter sites with too few subjects
    site_counts = df.groupby(config.site_column)[config.subject_column].nunique()
    valid_sites = site_counts[site_counts >= config.min_subjects_per_site].index
    df = df[df[config.site_column].isin(valid_sites)].copy()

    # Re-encode after filtering
    site_encoder = {s: i for i, s in enumerate(sorted(df[config.site_column].unique()))}
    site_names = sorted(df[config.site_column].unique())
    n_sites = len(site_names)

    # Encode subjects
    subject_encoder = {s: i for i, s in enumerate(sorted(df[config.subject_column].unique()))}
    n_subjects = len(subject_encoder)

    # Determine feature columns
    if config.feature_columns is not None:
        feature_cols = config.feature_columns
    else:
        exclude = {config.subject_column, config.site_column, config.target_column}
        feature_cols = [c for c in df.columns if c not in exclude]

    # Extract data
    features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    targets = torch.tensor(df[config.target_column].values, dtype=torch.float32)
    subject_ids = torch.tensor(
        [subject_encoder[s] for s in df[config.subject_column]], dtype=torch.long
    )
    site_ids = torch.tensor([site_encoder[s] for s in df[config.site_column]], dtype=torch.long)

    # Normalize
    if config.normalize:
        if config.normalize_mode == "per_site":
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

    return MRIData(
        features=features,
        targets=targets,
        subject_ids=subject_ids,
        site_ids=site_ids,
        n_sites=n_sites,
        n_subjects=n_subjects,
        feature_names=feature_cols,
        site_names=site_names,
        subject_site_matrix=subject_site_matrix,
    )


def create_ood_splits(
    mri_data: MRIData,
    held_out_site: int | str,
) -> tuple[MRIData, MRIData]:
    """Split data: train = all sites except held_out, test = held_out site only.

    Args:
        mri_data: Full dataset.
        held_out_site: Site ID (int) or name (str) to hold out.

    Returns:
        Tuple of (train_data, test_data).
    """
    if isinstance(held_out_site, str):
        held_out_idx = mri_data.site_names.index(held_out_site)
    else:
        held_out_idx = held_out_site

    test_mask = mri_data.site_ids == held_out_idx
    train_mask = ~test_mask

    return _subset(mri_data, train_mask), _subset(mri_data, test_mask)


def create_transfer_splits(
    mri_data: MRIData,
    target_site: int | str,
    n_target_samples: int,
    seed: int = 42,
) -> tuple[MRIData, MRIData, MRIData]:
    """Split for few-shot transfer evaluation.

    Source = all sites except the target. The target site is split into a
    finetune subset of n_target_samples and a disjoint test subset, so no
    target sample appears in both.

    Args:
        mri_data: Full dataset.
        target_site: Site ID (int) or name (str) used as transfer target.
        n_target_samples: Number of target samples for finetuning (capped
            so at least one test sample remains).
        seed: Seed for the finetune/test permutation.

    Returns:
        Tuple of (source_train, target_finetune, target_test).
    """
    if isinstance(target_site, str):
        target_idx = mri_data.site_names.index(target_site)
    else:
        target_idx = target_site

    target_mask = mri_data.site_ids == target_idx
    source_mask = ~target_mask

    # Split target into finetune and test
    target_indices = torch.where(target_mask)[0]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(target_indices), generator=gen)

    n_ft = min(n_target_samples, len(target_indices) - 1)  # Keep at least 1 for test
    ft_indices = target_indices[perm[:n_ft]]
    test_indices = target_indices[perm[n_ft:]]

    ft_mask = torch.zeros(len(mri_data.site_ids), dtype=torch.bool)
    ft_mask[ft_indices] = True
    test_mask = torch.zeros(len(mri_data.site_ids), dtype=torch.bool)
    test_mask[test_indices] = True

    return _subset(mri_data, source_mask), _subset(mri_data, ft_mask), _subset(mri_data, test_mask)


def _subset(mri_data: MRIData, mask: torch.Tensor) -> MRIData:
    """Create a subset of MRIData using a boolean mask.

    Subject rows are re-indexed to the subset, but subject_ids values are
    kept as the original ids so splits remain comparable.
    """
    sub_ids = mri_data.subject_ids[mask]
    site_ids = mri_data.site_ids[mask]

    # Recompute subject-site matrix for the subset
    unique_subjects = sub_ids.unique()
    n_subjects_sub = len(unique_subjects)

    # Keep the original site count so site indices stay valid across splits.
    subject_site_matrix = torch.zeros(n_subjects_sub, mri_data.n_sites)
    sub_map = {int(s): i for i, s in enumerate(unique_subjects.tolist())}
    for i in range(len(sub_ids)):
        new_sub = sub_map[int(sub_ids[i].item())]
        subject_site_matrix[new_sub, site_ids[i]] = 1.0

    return MRIData(
        features=mri_data.features[mask],
        targets=mri_data.targets[mask],
        subject_ids=sub_ids,
        site_ids=site_ids,
        n_sites=mri_data.n_sites,
        n_subjects=n_subjects_sub,
        feature_names=mri_data.feature_names,
        site_names=mri_data.site_names,
        subject_site_matrix=subject_site_matrix,
    )


def mri_to_multienv_format(
    mri_data: MRIData,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert MRI data to (N_subjects, E, d) format using pseudobulk.

    For subjects in only one site, uses site-level pseudobulk averages
    for the missing sites.

    Returns:
        Tuple of:
        - observations: (N_subjects, E, d) tensor, pseudobulk-filled
        - mask: (N_subjects, E) boolean, True where subject has real data
    """
    n_sites = mri_data.n_sites
    d = mri_data.features.shape[1]
    unique_subjects = mri_data.subject_ids.unique()
    n_sub = len(unique_subjects)
    sub_map = {int(s): i for i, s in enumerate(unique_subjects.tolist())}

    # Compute per-site pseudobulk (mean across all subjects at that site)
    pseudobulk = torch.zeros(n_sites, d)
    site_counts = torch.zeros(n_sites)
    for i in range(len(mri_data.site_ids)):
        s = int(mri_data.site_ids[i].item())
        pseudobulk[s] += mri_data.features[i]
        site_counts[s] += 1

    for s in range(n_sites):
        if site_counts[s] > 0:
            pseudobulk[s] /= site_counts[s]

    # Build multi-env tensor
    observations = pseudobulk.unsqueeze(0).expand(n_sub, -1, -1).clone()
    mask = torch.zeros(n_sub, n_sites, dtype=torch.bool)

    for i in range(len(mri_data.subject_ids)):
        sub_idx = sub_map[int(mri_data.subject_ids[i].item())]
        site_idx = int(mri_data.site_ids[i].item())
        observations[sub_idx, site_idx] = mri_data.features[i]
        mask[sub_idx, site_idx] = True

    return observations, mask


def generate_synthetic_mri(
    n_subjects: int = 500,
    n_sites: int = 4,
    n_features: int = 100,
    n_classes: int = 2,
    site_effect_strength: float = 1.0,
    cross_site_fraction: float = 0.0,
    seed: int = 42,
) -> MRIData:
    """Generate synthetic multi-site MRI data with known site effects.

    Model: features_i = base_i + site_effect_s + noise
    where base_i depends on diagnosis (class label).

    Args:
        n_subjects: Total number of subjects.
        n_sites: Number of sites.
        n_features: Feature dimensionality.
        n_classes: Number of diagnostic classes.
        site_effect_strength: Magnitude of site-specific shifts.
        cross_site_fraction: Fraction of subjects scanned at multiple sites.
        seed: Random seed.

    Returns:
        MRIData with synthetic data.
    """
    gen = torch.Generator().manual_seed(seed)

    # Assign subjects to sites
    site_assignments = torch.randint(0, n_sites, (n_subjects,), generator=gen)
    targets = torch.randint(0, n_classes, (n_subjects,), generator=gen)

    # Class-specific base features
    class_centers = torch.randn(n_classes, n_features, generator=gen)

    # Site-specific effects
    site_effects = site_effect_strength * torch.randn(n_sites, n_features, generator=gen)

    # Generate features
    features_list = []
    targets_list = []
    subject_ids_list = []
    site_ids_list = []

    for i in range(n_subjects):
        site = int(site_assignments[i].item())
        cls = int(targets[i].item())
        feat = (
            class_centers[cls] + site_effects[site] + 0.5 * torch.randn(n_features, generator=gen)
        )
        features_list.append(feat)
        targets_list.append(targets[i])
        subject_ids_list.append(i)
        site_ids_list.append(site)

    # Add cross-site subjects
    n_cross = int(n_subjects * cross_site_fraction)
    for _j in range(n_cross):
        # Pick a random existing subject, scan at a different site
        orig_sub = torch.randint(0, n_subjects, (1,), generator=gen).item()
        orig_site = int(site_assignments[orig_sub].item())
        new_site = (
            orig_site + 1 + torch.randint(0, n_sites - 1, (1,), generator=gen).item()
        ) % n_sites
        cls = int(targets[orig_sub].item())
        feat = (
            class_centers[cls]
            + site_effects[new_site]
            + 0.5 * torch.randn(n_features, generator=gen)
        )
        features_list.append(feat)
        targets_list.append(targets[orig_sub])
        subject_ids_list.append(orig_sub)
        site_ids_list.append(new_site)

    features = torch.stack(features_list)
    all_targets = torch.tensor(
        [t.item() if isinstance(t, torch.Tensor) else t for t in targets_list], dtype=torch.float32
    )
    subject_ids = torch.tensor(subject_ids_list, dtype=torch.long)
    site_ids = torch.tensor(site_ids_list, dtype=torch.long)

    # Build subject-site matrix
    subject_site_matrix = torch.zeros(n_subjects, n_sites)
    for i in range(len(subject_ids)):
        subject_site_matrix[subject_ids[i], site_ids[i]] = 1.0

    feature_names = [f"feature_{i}" for i in range(n_features)]
    site_names = [f"Site_{i}" for i in range(n_sites)]

    return MRIData(
        features=features,
        targets=all_targets,
        subject_ids=subject_ids,
        site_ids=site_ids,
        n_sites=n_sites,
        n_subjects=n_subjects,
        feature_names=feature_names,
        site_names=site_names,
        subject_site_matrix=subject_site_matrix,
    )
