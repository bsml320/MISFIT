"""Program dictionary: structured SVD parameterization of program operators.

Each program B_k = U_k @ diag(sigma_k) @ V_k^T where U_k, V_k are
semi-orthogonal and sigma_k are unconstrained singular values.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ProgramDictionary(nn.Module):
    """Dictionary of K learnable program operators B_k.

    Parameterized via truncated SVD:
        B_k = U_k @ diag(sigma_k) @ V_k^T

    Args:
        n_programs: Number of programs K.
        dim: Representation dimension d.
        rank: Rank of each program operator. None = full rank (d).
    """

    def __init__(
        self,
        n_programs: int,
        dim: int,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        self.n_programs = n_programs
        self.dim = dim
        self.rank = rank if rank is not None else dim

        # U_k: (K, d, rank), V_k: (K, d, rank), sigma_k: (K, rank)
        self.U = nn.Parameter(torch.empty(n_programs, dim, self.rank))
        self.V = nn.Parameter(torch.empty(n_programs, dim, self.rank))
        self.sigma = nn.Parameter(torch.empty(n_programs, self.rank))

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize with small random orthogonal U, V and unit sigma."""
        for k in range(self.n_programs):
            U_k = torch.randn(self.dim, self.rank)
            V_k = torch.randn(self.dim, self.rank)
            U_k, _ = torch.linalg.qr(U_k)
            V_k, _ = torch.linalg.qr(V_k)
            self.U.data[k] = U_k[:, : self.rank]
            self.V.data[k] = V_k[:, : self.rank]
        nn.init.ones_(self.sigma)
        self.sigma.data *= 0.01  # start near zero operator

    def forward(self) -> torch.Tensor:
        """Compute all K program operators.

        Returns:
            Shape (K, d, d) tensor of program operators B_k.
        """
        # B_k = U_k @ diag(sigma_k) @ V_k^T
        # (K, d, rank) @ (K, rank, rank) @ (K, rank, d)
        S = torch.diag_embed(self.sigma)  # (K, rank, rank)
        return self.U @ S @ self.V.transpose(-2, -1)

    def get_program(self, k: int) -> torch.Tensor:
        """Get single program operator B_k.

        Args:
            k: Program index.

        Returns:
            Shape (d, d) program operator.
        """
        S_k = torch.diag(self.sigma[k])  # (rank, rank)
        return self.U[k] @ S_k @ self.V[k].T

    def get_directions(self) -> torch.Tensor:
        """Get leading right singular vector of each program.

        Returns:
            Shape (K, d) program directions (unit vectors).
        """
        # Leading right singular vector = V_k[:, argmax(|sigma_k|)]
        idx = self.sigma.abs().argmax(dim=1)  # (K,)
        dirs = torch.stack([self.V[k, :, idx[k]] for k in range(self.n_programs)])
        return dirs / dirs.norm(dim=1, keepdim=True)

    @classmethod
    def init_from_spectral(
        cls,
        directions: torch.Tensor,
        dim: int | None = None,
        rank: int = 1,
    ) -> ProgramDictionary:
        """Initialize from tensor spectral recovery output.

        Each program is initialized as rank-1: B_k = b_k @ b_k^T.

        Args:
            directions: Shape (K, d) recovered program directions.
            dim: Representation dimension. If None, inferred from directions.
            rank: Rank of the operator parameterization.

        Returns:
            ProgramDictionary initialized from spectral recovery.
        """
        K, d = directions.shape
        if dim is None:
            dim = d

        prog = cls(n_programs=K, dim=dim, rank=rank)

        with torch.no_grad():
            for k in range(K):
                b_k = directions[k]
                B_k = torch.outer(b_k, b_k)  # rank-1: b_k @ b_k^T
                U_full, S_full, Vt_full = torch.linalg.svd(B_k, full_matrices=False)
                prog.U.data[k] = U_full[:, :rank]
                prog.sigma.data[k] = S_full[:rank]
                prog.V.data[k] = Vt_full[:rank, :].T

        return prog

    def export_artifacts(self, path: str | Path) -> None:
        """Save state dict and program directions for downstream phases.

        Args:
            path: Directory to save artifacts.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "program_dict_state.pt")
        torch.save(self.get_directions(), path / "program_directions.pt")

    @classmethod
    def load_from_phase1a(
        cls,
        path: str | Path,
        n_programs: int,
        dim: int,
        rank: int | None = None,
    ) -> ProgramDictionary:
        """Load from exported Phase 1A artifacts.

        Args:
            path: Directory containing exported artifacts.
            n_programs: Number of programs K.
            dim: Representation dimension d.
            rank: Rank of operator parameterization.

        Returns:
            ProgramDictionary with loaded weights.
        """
        path = Path(path)
        prog = cls(n_programs=n_programs, dim=dim, rank=rank)
        state = torch.load(path / "program_dict_state.pt", weights_only=True)
        prog.load_state_dict(state)
        return prog
