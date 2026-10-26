"""Hard negative mining for contrastive learning."""

from __future__ import annotations

import torch


def mine_hard_negatives(
    features: torch.Tensor,
    subject_ids: torch.Tensor,
    n_hard: int = 16,
) -> torch.Tensor:
    """Select hard negatives: closest samples from different subjects.

    For each sample, finds the top-n_hard most similar samples that belong
    to different subjects.

    Args:
        features: L2-normalized features, shape (batch, d).
        subject_ids: Subject label for each sample, shape (batch,).
        n_hard: Number of hard negatives per sample.

    Returns:
        Indices of hard negatives, shape (batch, n_hard).
    """
    n = features.shape[0]
    n_hard = min(n_hard, n - 1)

    # Similarity matrix
    sim = features @ features.T  # (n, n)

    # Mask out same-subject entries (includes self): -2.0 is below the
    # cosine-similarity range [-1, 1], so they never rank in the top-k.
    same_subject = subject_ids.unsqueeze(0) == subject_ids.unsqueeze(1)
    sim = sim.masked_fill(same_subject, -2.0)

    # Top-k most similar negatives
    _, indices = sim.topk(n_hard, dim=1)
    return indices
