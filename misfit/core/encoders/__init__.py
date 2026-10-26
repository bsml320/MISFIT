"""Encoder modules for FIRE-MECHANIST.

Exports BaseEncoder (abstract interface: forward(x) -> representation,
output_dim property) and MLPEncoder (feed-forward implementation).
"""

from misfit.core.encoders.base import BaseEncoder
from misfit.core.encoders.mlp import MLPEncoder

__all__ = ["BaseEncoder", "MLPEncoder"]
