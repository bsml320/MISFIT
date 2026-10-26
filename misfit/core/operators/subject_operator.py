"""Subject operator: T_i = I + sum_k a_{ik} B_k."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from misfit.core.operators.program_dict import ProgramDictionary


class SubjectOperator(nn.Module):
    """Applies subject-specific operator T_i to representations.

    T_i = I + sum_{k=1}^{K} a_{ik} B_k

    where a_{ik} are the subject's program loadings and B_k are learned
    program operators from ProgramDictionary.

    Args:
        program_dict: ProgramDictionary instance providing B_k operators.
    """

    def __init__(self, program_dict: ProgramDictionary) -> None:
        super().__init__()
        self.program_dict = program_dict

    def forward(
        self,
        loadings: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Apply T_i = I + sum_k a_{ik} B_k to representations.

        Args:
            loadings: Subject loadings of shape (batch, K).
            z: Representations of shape (batch, d).

        Returns:
            Transformed representations T_i @ z of shape (batch, d).
        """
        # B_k: (K, d, d)
        B = self.program_dict()

        # Weighted sum: sum_k a_{ik} B_k -> (batch, d, d)
        # loadings: (batch, K), B: (K, d, d) -> einsum -> (batch, d, d)
        weighted_B = torch.einsum("bk,kij->bij", loadings, B)

        # T_i @ z = (I + weighted_B) @ z = z + weighted_B @ z
        return z + torch.einsum("bij,bj->bi", weighted_B, z)

    @staticmethod
    def export_loadings(
        loadings: torch.Tensor,
        save_dir: str | Path,
        subject_ids: torch.Tensor | None = None,
        trainable: bool = False,
    ) -> Path:
        """Export per-subject loadings to ``subject_loadings.pt`` (Phase 1A artifact).

        Writes the loadings the downstream phases consume, with metadata (shapes,
        whether they were optimized in Stage 2). See PHASE.md export contract.

        Args:
            loadings: Per-subject loadings of shape (n_subjects, K).
            save_dir: Directory to write ``subject_loadings.pt`` into.
            subject_ids: Optional (n_subjects,) row->subject id mapping.
            trainable: Whether the loadings were optimized in Stage 2 (vs held fixed).

        Returns:
            Path to the written file.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "subject_loadings.pt"
        torch.save(
            {
                "loadings": loadings.detach().cpu(),
                "subject_ids": None if subject_ids is None else subject_ids.detach().cpu(),
                "n_subjects": int(loadings.shape[0]),
                "n_programs": int(loadings.shape[1]),
                "trainable": bool(trainable),
            },
            out,
        )
        return out
