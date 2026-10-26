"""Two-stage trainer for FIRE operator model.

Stage 1: Spectral initialization via tensor recovery.
Stage 2: End-to-end gradient refinement with combined loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from misfit.core.contrastive.losses import supervised_contrastive_loss
from misfit.core.contrastive.pairs import build_cross_env_pairs
from misfit.core.contrastive.projection import ProjectionHead
from misfit.core.losses import LossBreakdown, total_loss
from misfit.core.operators.operator_model import OperatorModel
from misfit.core.operators.program_dict import ProgramDictionary
from misfit.core.optim import EarlyStopping
from misfit.core.types import TrainConfig
from misfit.tensor_spectral.recovery import RecoveryResult

logger = logging.getLogger("fire_mechanist")


@dataclass
class TrainMetrics:
    """Metrics from a training run."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    loss_components: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    stage1_time: float = 0.0
    stage2_time: float = 0.0


class TwoStageTrainer:
    """Two-stage trainer: spectral init + gradient refinement.

    Stage 1: Initialize ProgramDictionary from tensor spectral recovery.
    Stage 2: Train full OperatorModel end-to-end with combined loss.

    Args:
        model: OperatorModel to train.
        config: Training configuration.
        task: "regression" or "classification".
        use_contrastive: Whether to include contrastive loss.
        contrastive_dim: Projection head output dimension for contrastive.
    """

    def __init__(
        self,
        model: OperatorModel,
        config: TrainConfig | None = None,
        task: str = "regression",
        use_contrastive: bool = True,
        contrastive_dim: int = 64,
    ) -> None:
        self.model = model
        self.config = config or TrainConfig()
        self.task = task
        self.use_contrastive = use_contrastive

        self.projection_head: ProjectionHead | None = None
        if use_contrastive:
            dim = model.encoder.output_dim
            self.projection_head = ProjectionHead(
                input_dim=dim, hidden_dim=128, output_dim=contrastive_dim
            )

        self.metrics = TrainMetrics()

    def stage1_init(self, recovery_result: RecoveryResult) -> None:
        """Initialize ProgramDictionary from spectral recovery.

        Args:
            recovery_result: Result from recover_programs().
        """
        t0 = time.time()
        directions = recovery_result.directions.detach()
        prog = self.model.program_dict

        new_prog = ProgramDictionary.init_from_spectral(directions, dim=prog.dim, rank=prog.rank)

        # Copy weights
        with torch.no_grad():
            prog.U.data.copy_(new_prog.U.data)
            prog.V.data.copy_(new_prog.V.data)
            prog.sigma.data.copy_(new_prog.sigma.data)

        self.metrics.stage1_time = time.time() - t0
        logger.info("Stage 1: Initialized %d programs from spectral recovery.", prog.n_programs)

    def stage2_train(
        self,
        train_data: dict[str, torch.Tensor],
        val_data: dict[str, torch.Tensor] | None = None,
    ) -> TrainMetrics:
        """Run Stage 2 end-to-end training.

        Args:
            train_data: Dict with keys "x", "targets", "loadings",
                "env_ids", and optionally "subject_ids".
            val_data: Same structure for validation. If None, uses train_data.

        Returns:
            TrainMetrics with training history.
        """
        t0 = time.time()
        cfg = self.config
        if cfg.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(cfg.device)

        self.model.to(device)
        if self.projection_head is not None:
            self.projection_head.to(device)

        # Freeze programs if configured
        if cfg.freeze_programs:
            for p in self.model.program_dict.parameters():
                p.requires_grad_(False)
            logger.info("Stage 2: Program parameters frozen.")

        # Optionally make the per-subject loadings {a_ik} trainable. Default off
        # keeps the original benchmark behavior (loadings held fixed at their input
        # values, L1 sparsity inactive). When on, a per-subject loading parameter is
        # created (indexed by subject_ids) and added to the optimizer.
        self._subject_loadings = None
        if cfg.optimize_loadings and "subject_ids" in train_data:
            sids = train_data["subject_ids"].long()
            self._subject_loadings = torch.nn.Parameter(
                torch.zeros(
                    int(sids.max().item()) + 1,
                    train_data["loadings"].shape[1],
                    device=device,
                )
            )
            with torch.no_grad():
                self._subject_loadings[sids] = train_data["loadings"].to(device)
            logger.info("Stage 2: per-subject loadings are trainable (optimize_loadings=True).")

        # Build optimizer with optional differential LR for programs
        prog_params = list(self.model.program_dict.parameters())
        prog_param_ids = {id(p) for p in prog_params}
        other_params = [p for p in self.model.parameters() if id(p) not in prog_param_ids]
        if self.projection_head is not None:
            other_params += list(self.projection_head.parameters())
        if self._subject_loadings is not None:
            other_params.append(self._subject_loadings)

        trainable_prog = [p for p in prog_params if p.requires_grad]
        if trainable_prog and cfg.program_lr is not None:
            param_groups = [
                {"params": other_params, "lr": cfg.learning_rate},
                {"params": trainable_prog, "lr": cfg.program_lr},
            ]
        else:
            param_groups = [{"params": other_params + trainable_prog, "lr": cfg.learning_rate}]

        optimizer = torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)

        # Build data loaders
        train_loader = self._build_loader(train_data, cfg.batch_size, shuffle=True)
        val_loader = None
        if val_data is not None:
            val_loader = self._build_loader(val_data, cfg.batch_size, shuffle=False)

        early_stop = EarlyStopping(patience=cfg.patience, mode="min")
        best_state = None

        for epoch in range(cfg.max_epochs):
            # Train
            self.model.train()
            if self.projection_head is not None:
                self.projection_head.train()

            epoch_loss = 0.0
            epoch_components: dict[str, float] = {}
            n_batches = 0

            for batch in train_loader:
                loss_bd = self._train_step(batch, optimizer, device)
                epoch_loss += loss_bd.total.item()
                for k, v in loss_bd.components.items():
                    epoch_components[k] = epoch_components.get(k, 0.0) + v
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            avg_components = {k: v / max(n_batches, 1) for k, v in epoch_components.items()}
            self.metrics.train_losses.append(avg_train_loss)
            self.metrics.loss_components.append(avg_components)

            # Validate
            if val_loader is not None:
                val_loss = self._validate(val_loader, device)
            else:
                val_loss = avg_train_loss
            self.metrics.val_losses.append(val_loss)

            if val_loss < self.metrics.best_val_loss:
                self.metrics.best_val_loss = val_loss
                self.metrics.best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if epoch % 10 == 0 or epoch == cfg.max_epochs - 1:
                logger.info("Epoch %d: train=%.4f, val=%.4f", epoch, avg_train_loss, val_loss)

            if early_stop.step(val_loss):
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(device)

        # Unfreeze programs if they were frozen
        if cfg.freeze_programs:
            for p in self.model.program_dict.parameters():
                p.requires_grad_(True)

        # Retain the per-subject loadings actually used (trainable param if
        # optimize_loadings, else the fixed inputs) so export_artifacts can write
        # subject_loadings.pt for downstream phases.
        if self._subject_loadings is not None:
            self._final_loadings = self._subject_loadings.detach().cpu()
        else:
            self._final_loadings = train_data["loadings"].detach().cpu()
        self._final_subject_ids = (
            train_data["subject_ids"].detach().cpu() if "subject_ids" in train_data else None
        )

        self.metrics.stage2_time = time.time() - t0
        return self.metrics

    def _train_step(
        self,
        batch: tuple[torch.Tensor, ...],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> LossBreakdown:
        """Single training step."""
        x, targets, loadings, env_ids = batch[:4]
        subject_ids = batch[4] if len(batch) > 4 else None

        x = x.to(device)
        targets = targets.to(device)
        env_ids = env_ids.to(device)
        # When loadings are trainable, index the optimized per-subject parameter
        # by subject_ids so gradients update the loadings; otherwise use the fixed
        # batch tensor unchanged.
        if getattr(self, "_subject_loadings", None) is not None and subject_ids is not None:
            loadings = self._subject_loadings[subject_ids.to(device).long()]
        else:
            loadings = loadings.to(device)

        optimizer.zero_grad()

        # Forward
        predictions = self.model(x, loadings, env_ids)

        # Content and style for separation loss
        z = self.model.encoder(x)
        content = self.model.subject_op(loadings, z)
        style = None
        if self.model.use_style:
            style = self.model.style_op(z, env_ids)

        # Contrastive loss: supervised NT-Xent that pulls same-subject representations
        # together across environments, using the true same-subject positives via
        # subject_ids. build_cross_env_pairs only gates on whether any cross-env
        # positive exists in the batch; arbitrary halves of a shuffled batch are
        # not real positives, so the loss is skipped when no true pair exists.
        contrastive_val = None
        if self.use_contrastive and subject_ids is not None and self.projection_head is not None:
            subject_ids = subject_ids.to(device)
            pairs = build_cross_env_pairs(content.detach(), subject_ids, env_ids)
            if pairs.anchor.shape[0] > 0:
                z_proj = torch.nn.functional.normalize(self.projection_head(content), dim=-1)
                contrastive_val = supervised_contrastive_loss(
                    z_proj,
                    subject_ids,
                    temperature=self.config.contrastive_temperature,
                )

        cfg = self.config
        loss_bd = total_loss(
            predictions=predictions,
            targets=targets,
            loadings=loadings,
            content=content,
            style=style,
            contrastive_loss_val=contrastive_val,
            task=self.task,
            sparsity_weight=cfg.sparsity_weight,
            style_weight=cfg.style_separation_weight,
            contrastive_weight=cfg.contrastive_weight,
        )

        loss_bd.total.backward()
        optimizer.step()

        return loss_bd

    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
        device: torch.device,
    ) -> float:
        """Compute validation loss."""
        self.model.eval()
        total = 0.0
        n = 0

        for batch in val_loader:
            x, targets, loadings, env_ids = batch[:4]
            x = x.to(device)
            targets = targets.to(device)
            loadings = loadings.to(device)
            env_ids = env_ids.to(device)

            predictions = self.model(x, loadings, env_ids)
            from misfit.core.losses import prediction_loss

            loss = prediction_loss(predictions, targets, self.task)
            total += loss.item() * x.shape[0]
            n += x.shape[0]

        return total / max(n, 1)

    def _build_loader(
        self,
        data: dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """Build DataLoader from data dict."""
        tensors = [data["x"], data["targets"], data["loadings"], data["env_ids"]]
        if "subject_ids" in data:
            tensors.append(data["subject_ids"])
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def export_artifacts(self, path: str | Path) -> None:
        """Export model and training artifacts.

        Args:
            path: Directory to save artifacts.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model_state.pt")
        self.model.program_dict.export_artifacts(path / "program_dict")
        # Export the per-subject loadings (subject_loadings.pt) required by later
        # phases. Available after stage2_train; skipped if training has not run.
        if getattr(self, "_final_loadings", None) is not None:
            self.model.subject_op.export_loadings(
                self._final_loadings,
                path,
                subject_ids=getattr(self, "_final_subject_ids", None),
                trainable=getattr(self, "_subject_loadings", None) is not None,
            )
        torch.save(
            {
                "train_losses": self.metrics.train_losses,
                "val_losses": self.metrics.val_losses,
                "best_epoch": self.metrics.best_epoch,
                "best_val_loss": self.metrics.best_val_loss,
                "stage1_time": self.metrics.stage1_time,
                "stage2_time": self.metrics.stage2_time,
            },
            path / "training_metrics.pt",
        )
