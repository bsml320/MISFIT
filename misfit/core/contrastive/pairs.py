"""Contrastive pair construction for cross-environment operator learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ContrastiveBatch:
    """A batch of contrastive pairs.

    Attributes:
        anchor: Anchor features of shape (n_pairs, d).
        positive: Positive features of shape (n_pairs, d).
        subject_ids: Subject indices for each pair, shape (n_pairs,).
        env_ids_anchor: Environment of anchor, shape (n_pairs,).
        env_ids_positive: Environment of positive, shape (n_pairs,).
    """

    anchor: torch.Tensor
    positive: torch.Tensor
    subject_ids: torch.Tensor
    env_ids_anchor: torch.Tensor
    env_ids_positive: torch.Tensor


def build_cross_env_pairs(
    features: torch.Tensor,
    subject_ids: torch.Tensor,
    env_ids: torch.Tensor,
) -> ContrastiveBatch:
    """Build positive pairs from same subject across different environments.

    For each subject observed in multiple environments, creates all pairwise
    combinations (e1, e2) with e1 < e2 as positive pairs.

    Args:
        features: Operator-transformed features of shape (batch, d).
        subject_ids: Subject index for each sample, shape (batch,).
        env_ids: Environment index for each sample, shape (batch,).

    Returns:
        ContrastiveBatch with cross-environment positive pairs.
    """
    device = features.device
    anchors, positives = [], []
    s_ids, env_a, env_p = [], [], []

    unique_subjects = subject_ids.unique()
    for s in unique_subjects:
        mask = subject_ids == s
        indices = mask.nonzero(as_tuple=True)[0]
        envs = env_ids[indices]

        # All index pairs ii < jj within the subject whose environments differ.
        for ii in range(len(indices)):
            for jj in range(ii + 1, len(indices)):
                if envs[ii] != envs[jj]:
                    anchors.append(features[indices[ii]])
                    positives.append(features[indices[jj]])
                    s_ids.append(s)
                    env_a.append(envs[ii])
                    env_p.append(envs[jj])

    if not anchors:
        # No pairs found; return empty batch
        d = features.shape[1]
        return ContrastiveBatch(
            anchor=torch.empty(0, d, device=device),
            positive=torch.empty(0, d, device=device),
            subject_ids=torch.empty(0, dtype=torch.long, device=device),
            env_ids_anchor=torch.empty(0, dtype=torch.long, device=device),
            env_ids_positive=torch.empty(0, dtype=torch.long, device=device),
        )

    return ContrastiveBatch(
        anchor=torch.stack(anchors),
        positive=torch.stack(positives),
        subject_ids=torch.stack(s_ids),
        env_ids_anchor=torch.stack(env_a),
        env_ids_positive=torch.stack(env_p),
    )
