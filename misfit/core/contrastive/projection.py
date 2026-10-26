"""Projection head for contrastive learning."""

from __future__ import annotations

import torch
from torch import nn


class ProjectionHead(nn.Module):
    """2-layer MLP projection head for contrastive embedding.

    Maps operator representations to a lower-dimensional space where
    contrastive loss is applied. Following SimCLR convention, this head
    is discarded after training.

    Args:
        input_dim: Dimension of input features.
        hidden_dim: Dimension of hidden layer.
        output_dim: Dimension of contrastive embedding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize.

        Args:
            x: Input features of shape (batch, input_dim).

        Returns:
            L2-normalized embeddings of shape (batch, output_dim).
        """
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)
