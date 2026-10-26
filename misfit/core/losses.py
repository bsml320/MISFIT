"""Loss functions for FIRE operator model training.

Provides the prediction loss (MSE or cross-entropy), sparsity penalties on
subject loadings, an HSIC-style content/style decorrelation term, and
total_loss, which combines them with configurable weights and returns a
LossBreakdown carrying each component for logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class LossBreakdown:
    """Breakdown of total loss into components."""

    total: torch.Tensor
    prediction: torch.Tensor
    sparsity: torch.Tensor
    style_separation: torch.Tensor
    contrastive: torch.Tensor
    components: dict[str, float] = field(default_factory=dict)


def prediction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task: str = "regression",
) -> torch.Tensor:
    """Compute prediction loss.

    Args:
        predictions: Model predictions, shape (batch, output_dim).
        targets: Ground truth, shape (batch,) or (batch, output_dim).
        task: "regression" (MSE) or "classification" (cross-entropy).

    Returns:
        Scalar loss.
    """
    if task == "regression":
        return F.mse_loss(predictions.squeeze(-1), targets.float())
    elif task == "classification":
        return F.cross_entropy(predictions, targets.long())
    else:
        raise ValueError(f"Unknown task '{task}'. Choose 'regression' or 'classification'.")


def sparsity_loss(
    loadings: torch.Tensor,
    sparsity_type: str = "l1",
) -> torch.Tensor:
    """Compute sparsity-inducing loss on loadings.

    Args:
        loadings: Subject loadings, shape (batch, K).
        sparsity_type: "l1" for L1 norm, "group" for group sparsity.

    Returns:
        Scalar loss.
    """
    if sparsity_type == "l1":
        return loadings.abs().mean()
    elif sparsity_type == "group":
        # Group sparsity: L2 norm per subject, then L1 across subjects
        return loadings.norm(dim=1).mean()
    else:
        raise ValueError(f"Unknown sparsity type '{sparsity_type}'.")


def style_separation_loss(
    content: torch.Tensor,
    style: torch.Tensor,
) -> torch.Tensor:
    """HSIC-based decorrelation between content and style representations.

    Computes a simplified HSIC estimate to encourage statistical independence
    between content (T_i z) and style (S_e z) representations.

    Args:
        content: Content features, shape (batch, d).
        style: Style features, shape (batch, d).

    Returns:
        Scalar loss (lower = more independent).
    """
    n = content.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=content.device, requires_grad=True)

    # Center
    content_c = content - content.mean(dim=0, keepdim=True)
    style_c = style - style.mean(dim=0, keepdim=True)

    # Cross-covariance matrix
    cross_cov = (content_c.T @ style_c) / (n - 1)  # (d, d)

    # Linear-kernel HSIC reduces to the squared Frobenius norm of the
    # cross-covariance; divide by d_content * d_style for scale invariance.
    return (cross_cov**2).sum() / (content.shape[1] * style.shape[1])


def total_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loadings: torch.Tensor,
    content: torch.Tensor | None = None,
    style: torch.Tensor | None = None,
    contrastive_loss_val: torch.Tensor | None = None,
    task: str = "regression",
    sparsity_type: str = "l1",
    sparsity_weight: float = 0.01,
    style_weight: float = 0.1,
    contrastive_weight: float = 0.1,
) -> LossBreakdown:
    """Compute combined training loss.

    L = L_pred + w1 * L_sparsity + w2 * L_style_sep + w3 * L_contrastive

    Args:
        predictions: Model predictions.
        targets: Ground truth targets.
        loadings: Subject loadings.
        content: Content features for style separation.
        style: Style features for style separation.
        contrastive_loss_val: Pre-computed contrastive loss.
        task: Task type.
        sparsity_type: Sparsity penalty type.
        sparsity_weight: Weight for sparsity loss.
        style_weight: Weight for style separation loss.
        contrastive_weight: Weight for contrastive loss.

    Returns:
        LossBreakdown with total and component losses.
    """
    device = predictions.device
    l_pred = prediction_loss(predictions, targets, task)
    l_sparse = sparsity_loss(loadings, sparsity_type)

    l_style = torch.tensor(0.0, device=device)
    if content is not None and style is not None:
        l_style = style_separation_loss(content, style)

    l_contrastive = torch.tensor(0.0, device=device)
    if contrastive_loss_val is not None:
        l_contrastive = contrastive_loss_val

    l_total = (
        l_pred
        + sparsity_weight * l_sparse
        + style_weight * l_style
        + contrastive_weight * l_contrastive
    )

    return LossBreakdown(
        total=l_total,
        prediction=l_pred,
        sparsity=l_sparse,
        style_separation=l_style,
        contrastive=l_contrastive,
        components={
            "prediction": l_pred.item(),
            "sparsity": l_sparse.item(),
            "style_separation": l_style.item(),
            "contrastive": l_contrastive.item(),
        },
    )
