"""Contrastive loss functions for operator learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def operator_contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """NT-Xent loss for cross-environment operator contrastive learning.

    Treats the anchor-positive pair as the positive example, and all other
    samples in the batch (both anchors and positives) as negatives.

    Args:
        anchor: L2-normalized anchor embeddings, shape (n_pairs, d).
        positive: L2-normalized positive embeddings, shape (n_pairs, d).
        temperature: Temperature scaling parameter.

    Returns:
        Scalar loss value.
    """
    n = anchor.shape[0]
    if n == 0:
        return torch.tensor(0.0, device=anchor.device, requires_grad=True)

    # Concatenate all embeddings: [anchor_0, ..., anchor_{n-1}, pos_0, ..., pos_{n-1}]
    embeddings = torch.cat([anchor, positive], dim=0)  # (2n, d)

    # Similarity matrix
    sim = embeddings @ embeddings.T / temperature  # (2n, 2n)

    # Mask out self-similarity
    mask_self = torch.eye(2 * n, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask_self, -1e9)

    # Positive pairs: anchor_i <-> positive_i
    # For anchor_i (index i), positive is at index i + n
    # For positive_i (index i + n), positive is at index i
    labels = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)], dim=0).to(sim.device)

    return F.cross_entropy(sim, labels)


def supervised_contrastive_loss(
    features: torch.Tensor,
    subject_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss (SupCon) using subject labels.

    All samples from the same subject are positives; different subjects
    are negatives.

    Args:
        features: L2-normalized features, shape (batch, d).
        subject_ids: Subject label for each sample, shape (batch,).
        temperature: Temperature scaling parameter.

    Returns:
        Scalar loss value.
    """
    n = features.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    # Similarity matrix
    sim = features @ features.T / temperature  # (n, n)

    # Mask: 1 if same subject, 0 otherwise
    label_mask = subject_ids.unsqueeze(0) == subject_ids.unsqueeze(1)  # (n, n)
    # Remove self-comparisons
    self_mask = ~torch.eye(n, device=features.device, dtype=torch.bool)
    positive_mask = label_mask & self_mask

    # If no positive pairs exist, return 0
    if not positive_mask.any():
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    # For numerical stability
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    sim = sim - sim_max

    # Denominator: sum over all non-self samples
    exp_sim = torch.exp(sim) * self_mask.float()
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-10)

    # Numerator: log-sum over positive pairs
    log_prob = sim - log_denom  # (n, n)

    # Average over positive pairs per sample
    n_positives = positive_mask.float().sum(dim=1)  # (n,)
    has_positive = n_positives > 0

    mean_log_prob = (log_prob * positive_mask.float()).sum(dim=1) / (n_positives + 1e-10)

    loss = -mean_log_prob[has_positive].mean()
    return loss
