"""Full FIRE operator model: encoder -> T_i -> prediction."""

from __future__ import annotations

import torch
from torch import nn

from misfit.core.encoders.base import BaseEncoder
from misfit.core.operators.program_dict import ProgramDictionary
from misfit.core.operators.style_operator import StyleOperator
from misfit.core.operators.subject_operator import SubjectOperator


class OperatorModel(nn.Module):
    """Full FIRE operator model.

    Forward pass:
        z = encoder(x)
        content = T_i @ z   where T_i = I + sum_k a_{ik} B_k
        style = S_e @ z     (optional)
        output = head(cat(content, style)) or head(content)

    Args:
        encoder: BaseEncoder mapping observations to representations.
        program_dict: ProgramDictionary providing program operators B_k.
        n_envs: Number of environments.
        output_dim: Dimension of prediction output.
        use_style: Whether to include style operator.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        program_dict: ProgramDictionary,
        n_envs: int,
        output_dim: int,
        use_style: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.subject_op = SubjectOperator(program_dict)
        self.program_dict = program_dict
        self.use_style = use_style

        dim = encoder.output_dim
        head_input_dim = dim

        if use_style:
            self.style_op = StyleOperator(n_envs=n_envs, dim=dim)
            head_input_dim = 2 * dim

        self.head = nn.Linear(head_input_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        loadings: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass.

        Args:
            x: Input observations of shape (batch, input_dim).
            loadings: Subject loadings of shape (batch, K).
            env_ids: Environment indices of shape (batch,), integer.

        Returns:
            Predictions of shape (batch, output_dim).
        """
        z = self.encoder(x)
        content = self.subject_op(loadings, z)

        if self.use_style:
            style = self.style_op(z, env_ids)
            combined = torch.cat([content, style], dim=-1)
        else:
            combined = content

        return self.head(combined)
