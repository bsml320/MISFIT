"""MLP encoder: feed-forward implementation of BaseEncoder.

Provides MLPEncoder, the default encoder for vector-valued inputs
(ROI features, PCA scores). Builds a stack of Linear + activation
(+ optional Dropout) blocks followed by a linear output projection.
"""

from __future__ import annotations

import torch
from torch import nn

from misfit.core.encoders.base import BaseEncoder


class MLPEncoder(BaseEncoder):
    """Multi-layer perceptron encoder.

    Maps input observations to a d-dimensional representation via a
    feed-forward network with configurable depth, width, and activation.

    Args:
        input_dim: Dimension of input observations.
        hidden_dims: List of hidden layer dimensions.
        repr_dim: Dimension of output representation.
        activation: Activation function name ("relu", "gelu", "tanh").
        dropout: Dropout probability between layers.
    """

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        repr_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose from {list(self._ACTIVATIONS.keys())}."
            )

        self._repr_dim = repr_dim
        act_cls = self._ACTIVATIONS[activation]

        # Hidden stack: Linear -> activation (-> Dropout) per hidden width.
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Output projection is linear: no activation or dropout on the
        # representation, so downstream operators act on an unbounded space.
        layers.append(nn.Linear(prev_dim, repr_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to representation.

        Args:
            x: Shape (batch, input_dim).

        Returns:
            Shape (batch, repr_dim).
        """
        return self.net(x)

    @property
    def output_dim(self) -> int:
        """Dimension of the output representation."""
        return self._repr_dim
