"""Numerically stable whitening of multi-environment observations.

Uses truncated eigendecomposition of second-order moments with condition
number monitoring. All moment accumulation is done in float64 to avoid
catastrophic cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch

logger = logging.getLogger("fire_mechanist")


@dataclass
class WhiteningResult:
    """Result of whitening computation."""

    whitening_matrices: dict[int, torch.Tensor]  # env_id -> W_e of shape (d, r_e)
    effective_ranks: dict[int, int]  # env_id -> r_e
    condition_numbers: dict[int, float]  # env_id -> kappa before whitening
    eigenvalues: dict[int, torch.Tensor]  # env_id -> full eigenvalue spectrum
    truncation_threshold: float


def compute_cross_env_whitening(
    observations: torch.Tensor,
    n_programs: int,
    truncation_threshold: float = 1e-6,
) -> WhiteningResult:
    """Compute whitening using cross-environment moments (noise-free).

    Two-step approach per environment:
    1. Cross-env SVD identifies the K-dim signal subspace without noise bias
       (M_{e,e'} = A_e C A_{e'}^T has no sigma^2 I term for e != e').
    2. Auto-moment whitening within the projected subspace isotropizes.

    This handles non-identity environment transforms A_e where the standard
    auto-moment whitening fails because noise corrupts the signal subspace.

    Args:
        observations: Tensor of shape (N, E, d).
        n_programs: Number of latent programs K (determines subspace rank).
        truncation_threshold: Eigenvalues below tau * max_eigenvalue are discarded.

    Returns:
        WhiteningResult containing whitening matrices and diagnostics.
    """
    N, E, d = observations.shape
    # Cross-env whitening averages between-environment cross-moments, so it needs
    # at least two environments. Fail with a clear message rather than dividing by
    # zero pairs and hitting an opaque non-finite SVD error downstream.
    if E < 2:
        raise ValueError(
            f"compute_cross_env_whitening requires E>=2 environments; got E={E}. "
            "Use compute_whitening (auto-moment) for single-environment data."
        )
    orig_dtype = observations.dtype
    orig_device = observations.device

    obs_f64 = observations.to(dtype=torch.float64)

    whitening_matrices: dict[int, torch.Tensor] = {}
    effective_ranks: dict[int, int] = {}
    condition_numbers: dict[int, float] = {}
    eigenvalue_dict: dict[int, torch.Tensor] = {}

    for e in range(E):
        # Step 1: Cross-env moments to find signal subspace (noise-free).
        # Average M_{e,e'} over all e' != e for variance reduction.
        # M_{e,e'} = A_e C A_{e'}^T has column space = span(A_e B).
        X_e = obs_f64[:, e, :]  # (N, d)
        M_cross = torch.zeros(d, d, dtype=torch.float64, device=orig_device)
        n_pairs = 0
        for ep in range(E):
            if ep == e:
                continue
            X_ep = obs_f64[:, ep, :]  # (N, d)
            M_cross = M_cross + (X_e.T @ X_ep) / N
            n_pairs += 1
        M_cross = M_cross / n_pairs

        # SVD of cross-env moment to extract signal subspace
        U, S, _Vh = torch.linalg.svd(M_cross, full_matrices=False)

        # Keep top K components (signal subspace of env e)
        K = min(n_programs, d, N)
        U_K = U[:, :K]  # (d, K) basis for signal subspace

        # Log cross-env singular value spectrum for diagnostics
        cross_sv_ratio = (S[0] / S[min(K, len(S)) - 1]).item() if len(S) >= K else float("inf")
        if cross_sv_ratio > 1e6:
            logger.warning(
                "Env %d: cross-env singular value ratio %.2e (signal may be ill-conditioned).",
                e, cross_sv_ratio,
            )

        # Step 2: Project data into K-dim subspace, then whiten within it.
        # Y_e = X_e @ U_K is (N, K). Auto-moment G = Y^T Y / N has noise
        # only as uniform shift sigma^2 I_K (since U_K spans signal subspace),
        # so eigenvector estimation is unbiased.
        Y_e = X_e @ U_K  # (N, K)
        G = (Y_e.T @ Y_e) / N  # (K, K)

        # Eigendecomposition of G (symmetric, ascending by torch)
        eigvals, eigvecs = torch.linalg.eigh(G)
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)

        eigenvalue_dict[e] = eigvals.to(dtype=orig_dtype)

        # Condition number
        max_eigval = eigvals[0].item()
        min_eigval = eigvals[-1].item()
        kappa = max_eigval / max(abs(min_eigval), 1e-30)
        condition_numbers[e] = kappa

        if kappa > 1e6:
            logger.warning(
                "Environment %d has condition number %.2e in projected subspace.", e, kappa
            )

        # Truncate: keep eigenvalues > tau * max_eigenvalue
        threshold = truncation_threshold * max_eigval
        keep_mask = eigvals > threshold
        r_e = int(keep_mask.sum().item())

        if r_e == 0:
            raise ValueError(
                f"Environment {e}: all projected eigenvalues below truncation threshold. "
                f"Max eigenvalue: {max_eigval:.2e}, threshold: {threshold:.2e}"
            )

        effective_ranks[e] = r_e

        if r_e < K:
            logger.info(
                "Environment %d: effective rank %d / %d in projected subspace.", e, r_e, K
            )

        # Full whitening: W_e = U_K @ Q[:, :r] @ diag(1/sqrt(Lambda[:r]))
        # Maps from d-dim original space to r-dim whitened signal space
        Q_r = eigvecs[:, :r_e]  # (K, r_e)
        inv_sqrt_lambda = 1.0 / torch.sqrt(eigvals[:r_e])  # (r_e,)
        W_e = U_K @ Q_r @ torch.diag(inv_sqrt_lambda)  # (d, r_e)

        whitening_matrices[e] = W_e.to(dtype=orig_dtype)

    return WhiteningResult(
        whitening_matrices=whitening_matrices,
        effective_ranks=effective_ranks,
        condition_numbers=condition_numbers,
        eigenvalues=eigenvalue_dict,
        truncation_threshold=truncation_threshold,
    )


def compute_whitening(
    observations: torch.Tensor,
    truncation_threshold: float = 1e-6,
    regularization: float = 0.0,
) -> WhiteningResult:
    """Compute whitening matrices for all environments.

    Args:
        observations: Tensor of shape (N, E, d).
        truncation_threshold: Eigenvalues below tau * max_eigenvalue are discarded.
        regularization: Optional Tikhonov regularization added to diagonal.

    Returns:
        WhiteningResult containing whitening matrices and diagnostics.
    """
    N, E, d = observations.shape
    orig_dtype = observations.dtype
    orig_device = observations.device

    # Accumulate in float64
    obs_f64 = observations.to(dtype=torch.float64)

    whitening_matrices: dict[int, torch.Tensor] = {}
    effective_ranks: dict[int, int] = {}
    condition_numbers: dict[int, float] = {}
    eigenvalue_dict: dict[int, torch.Tensor] = {}

    for e in range(E):
        # Compute second-order auto-moment: M_{e,e} = (1/N) * X_e^T X_e
        X_e = obs_f64[:, e, :]  # (N, d)
        M_ee = (X_e.T @ X_e) / N  # (d, d)

        # Add regularization
        if regularization > 0.0:
            M_ee = M_ee + regularization * torch.eye(d, dtype=torch.float64, device=orig_device)

        # Eigendecomposition (symmetric, sorted ascending by torch)
        eigvals, eigvecs = torch.linalg.eigh(M_ee)

        # Reverse to descending order
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)

        # Store full spectrum
        eigenvalue_dict[e] = eigvals.to(dtype=orig_dtype)

        # Condition number before whitening
        max_eigval = eigvals[0].item()
        min_eigval = eigvals[-1].item()
        kappa = max_eigval / max(abs(min_eigval), 1e-30)
        condition_numbers[e] = kappa

        if kappa > 1e6:
            logger.warning(
                "Environment %d has condition number %.2e before whitening.", e, kappa
            )

        # Truncate: keep eigenvalues > tau * max_eigenvalue
        threshold = truncation_threshold * max_eigval
        keep_mask = eigvals > threshold
        r_e = int(keep_mask.sum().item())

        if r_e == 0:
            raise ValueError(
                f"Environment {e}: all eigenvalues below truncation threshold. "
                f"Max eigenvalue: {max_eigval:.2e}, threshold: {threshold:.2e}"
            )

        effective_ranks[e] = r_e

        if r_e < d:
            logger.info(
                "Environment %d: effective rank %d / %d (truncated %d components).",
                e, r_e, d, d - r_e,
            )

        # Whitening matrix: W_e = U[:, :r] @ diag(1 / sqrt(Lambda[:r]))
        U_r = eigvecs[:, :r_e]  # (d, r_e)
        inv_sqrt_lambda = 1.0 / torch.sqrt(eigvals[:r_e])  # (r_e,)
        W_e = U_r @ torch.diag(inv_sqrt_lambda)  # (d, r_e)

        whitening_matrices[e] = W_e.to(dtype=orig_dtype)

    return WhiteningResult(
        whitening_matrices=whitening_matrices,
        effective_ranks=effective_ranks,
        condition_numbers=condition_numbers,
        eigenvalues=eigenvalue_dict,
        truncation_threshold=truncation_threshold,
    )


def apply_whitening(
    observations: torch.Tensor,
    whitening: WhiteningResult,
) -> torch.Tensor:
    """Apply whitening to observations.

    Args:
        observations: Shape (N, E, d).
        whitening: Result from compute_whitening.

    Returns:
        Whitened observations of shape (N, E, r_max) where r_max is
        the maximum effective rank across environments. Environments
        with lower rank are zero-padded.
    """
    N, E, d = observations.shape
    r_max = max(whitening.effective_ranks.values())

    whitened = torch.zeros(N, E, r_max, dtype=observations.dtype, device=observations.device)

    for e in range(E):
        W_e = whitening.whitening_matrices[e]  # (d, r_e)
        r_e = whitening.effective_ranks[e]
        # u_tilde = X_e @ W_e -> (N, r_e)
        whitened[:, e, :r_e] = observations[:, e, :] @ W_e

    return whitened


def compute_whitened_cross_moment(
    observations: torch.Tensor,
    whitening: WhiteningResult,
    env_indices: tuple[int, int, int],
    batch_size: int | None = None,
) -> torch.Tensor:
    """Compute cross-moment tensor of whitened observations.

    Args:
        observations: Shape (N, E, d).
        whitening: Result from compute_whitening.
        env_indices: Environment triple (e1, e2, e3).
        batch_size: Mini-batch size for memory efficiency. None = full batch.

    Returns:
        Whitened cross-moment tensor of shape (r, r, r) where r is the
        minimum effective rank across the triple.
    """
    N = observations.shape[0]
    e1, e2, e3 = env_indices

    W1 = whitening.whitening_matrices[e1]  # (d, r1)
    W2 = whitening.whitening_matrices[e2]  # (d, r2)
    W3 = whitening.whitening_matrices[e3]  # (d, r3)

    r = min(
        whitening.effective_ranks[e1],
        whitening.effective_ranks[e2],
        whitening.effective_ranks[e3],
    )

    # Truncate whitening matrices to common rank
    W1 = W1[:, :r]
    W2 = W2[:, :r]
    W3 = W3[:, :r]

    # Accumulate in float64
    obs_f64 = observations.to(dtype=torch.float64)
    W1_f64 = W1.to(dtype=torch.float64)
    W2_f64 = W2.to(dtype=torch.float64)
    W3_f64 = W3.to(dtype=torch.float64)

    T = torch.zeros(r, r, r, dtype=torch.float64, device=observations.device)

    if batch_size is None:
        batch_size = N

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        u1 = obs_f64[start:end, e1, :] @ W1_f64  # (B, r)
        u2 = obs_f64[start:end, e2, :] @ W2_f64  # (B, r)
        u3 = obs_f64[start:end, e3, :] @ W3_f64  # (B, r)

        # T += sum_i u1_i (x) u2_i (x) u3_i
        T += torch.einsum("bi,bj,bk->ijk", u1, u2, u3)

    T /= N

    return T.to(dtype=observations.dtype)
