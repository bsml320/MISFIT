"""Config dataclasses and shared type definitions for FIRE-MECHANIST."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TensorConfig:
    """Configuration for tensor spectral recovery."""

    n_subjects: int = 5000
    n_envs: int = 5
    n_programs: int = 3
    dim: int = 20
    noise_std: float = 0.1
    truncation_threshold: float = 1e-6
    n_restarts: int = 10
    max_power_iters: int = 1000
    power_tolerance: float = 1e-8


@dataclass
class OperatorConfig:
    """Configuration for operator modules."""

    n_programs: int = 3
    dim: int = 20
    operator_rank: int | None = None  # None = full rank (dim)
    sparsity_type: str = "l1"  # "l1" or "group"


@dataclass
class EncoderConfig:
    """Configuration for encoder networks."""

    input_dim: int = 100
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    output_dim: int = 20
    activation: str = "relu"
    dropout: float = 0.0


@dataclass
class TrainConfig:
    """Configuration for training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    sparsity_weight: float = 0.01
    style_separation_weight: float = 0.1
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.07
    freeze_programs: bool = False
    program_lr: float | None = None  # None = same as learning_rate
    # If True, Stage 2 also optimizes the per-subject loadings {a_ik} (added to the
    # optimizer, indexed by subject_ids). Default False preserves the frozen-loading
    # design of the Phase 1A NeurIPS submission (loadings held fixed; L1 sparsity
    # inactive). Enable only if you intend trainable loadings -- this changes results.
    optimize_loadings: bool = False
    seed: int = 42
    device: str = "auto"


@dataclass
class ExperimentConfig:
    """Configuration for a full experiment."""

    name: str = "default"
    train_config: TrainConfig = field(default_factory=TrainConfig)
    tensor_config: TensorConfig = field(default_factory=TensorConfig)
    operator_config: OperatorConfig = field(default_factory=OperatorConfig)
    n_seeds: int = 5
    output_dir: str = "runs"


@dataclass
class HierarchicalIOPConfig:
    """Configuration for the hierarchical IOP factorization.

    Models nested (entity, environment) structure:
        T_{i, c} = I + sum_k a_{ik} B_k + sum_l b_{cl} C_l

    where {B_k} are program operators across entities and {C_l} are
    environment-bias operators across environments.
    """

    K: int = 3
    L: int = 2
    d: int = 20
    n_entities: int = 100
    n_environments: int = 5
    frozen_programs: bool = True
    frozen_biases: bool = True
    rank: int | None = None


@dataclass
class ConformalConfig:
    """Configuration for split conformal selective prediction."""

    alpha: float = 0.1
    per_entity: bool = False
    min_calibration: int = 20


@dataclass
class MultiOmicsConfig:
    """Config for cross-modal identifiable multi-omics integration."""
    n_hvg: int = 2000
    n_pcs: int = 50
    n_programs: int = 10
    promoter_upstream: int = 2000     # bp upstream of TSS for cCRE->gene window
    min_cells_per_donor: int = 20
    cell_class: str = "Mic_Immune"
    seed: int = 0
