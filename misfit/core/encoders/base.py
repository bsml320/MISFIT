"""Abstract base encoder for FIRE-MECHANIST."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseEncoder(ABC, nn.Module):
    """Abstract base class for representation encoders.

    All encoders map input observations x to a d-dimensional representation z.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to representation.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Representation tensor of shape (batch, output_dim).
        """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of the output representation."""
