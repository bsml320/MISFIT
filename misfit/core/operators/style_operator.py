"""Style operator: environment-specific learnable transform."""

from __future__ import annotations

import torch
from torch import nn


class StyleOperator(nn.Module):
    """Environment-specific style transformation.

    Each environment e has a learnable linear transform S_e that captures
    environment-specific effects (scanner bias, protocol differences, etc.).

    Args:
        n_envs: Number of environments E.
        dim: Representation dimension d.
    """

    def __init__(self, n_envs: int, dim: int) -> None:
        super().__init__()
        self.n_envs = n_envs
        self.dim = dim

        # Initialize S_e as identity + small perturbation
        self.transforms = nn.Parameter(
            torch.eye(dim).unsqueeze(0).repeat(n_envs, 1, 1) + 0.01 * torch.randn(n_envs, dim, dim)
        )

    def forward(
        self,
        z: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply environment-specific style transform.

        Args:
            z: Representations of shape (batch, d).
            env_ids: Environment indices of shape (batch,), integer.

        Returns:
            Style-transformed representations S_e @ z of shape (batch, d).
        """
        # Gather per-sample transforms: (batch, d, d)
        S_batch = self.transforms[env_ids]
        return torch.einsum("bij,bj->bi", S_batch, z)

    def get_transform(self, env_id: int) -> torch.Tensor:
        """Get single environment's style transform.

        Args:
            env_id: Environment index.

        Returns:
            Shape (d, d) transform matrix.
        """
        return self.transforms[env_id]
