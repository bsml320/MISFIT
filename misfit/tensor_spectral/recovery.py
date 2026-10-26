"""Full tensor spectral recovery pipeline.

Integrates cross-moments, whitening, and tensor power method into a single
pipeline that takes raw multi-environment observations and returns estimated
program directions with full diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
import logging
import time

import torch

from misfit.core.types import TensorConfig
from misfit.tensor_spectral.tensor_power import (
    TensorDecompositionResult,
    tensor_decomposition,
)
from misfit.tensor_spectral.whitening import (
    WhiteningResult,
    compute_cross_env_whitening,
    compute_whitened_cross_moment,
    compute_whitening,
)

logger = logging.getLogger("fire_mechanist")


@dataclass
class RecoveryResult:
    """Full result of tensor spectral program recovery."""

    directions: torch.Tensor  # (K, d) recovered program directions
    eigenvalues: torch.Tensor  # (K,) eigenvalues from tensor decomposition
    whitening: WhiteningResult
    decomposition: TensorDecompositionResult
    diagnostics: dict = field(default_factory=dict)


def _select_env_triples(n_envs: int) -> list[tuple[int, int, int]]:
    """Select same-environment triples for symmetric tensor structure.

    Using (e, e, e) triples ensures the whitened cross-moment tensor is
    symmetric, which is required by the symmetric tensor power method.
    Averaging across environments improves signal-to-noise.

    Args:
        n_envs: Number of environments E.

    Returns:
        List of (e, e, e) triples, one per environment.
    """
    return [(e, e, e) for e in range(n_envs)]


def _symmetrize_tensor(T: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Force exact symmetry by averaging over all 6 permutations.

    Args:
        T: Tensor of shape (r, r, r).

    Returns:
        Tuple of (symmetrized tensor, max asymmetry error).
    """
    perms = list(permutations([0, 1, 2]))
    T_sym = torch.zeros_like(T)
    for p in perms:
        T_sym = T_sym + T.permute(p)
    T_sym = T_sym / len(perms)

    symmetry_error = (T - T_sym).abs().max().item()
    return T_sym, symmetry_error


def _hungarian_match(
    estimated: torch.Tensor,
    true: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    """Match estimated directions to true directions via greedy matching.

    Matches by maximum absolute cosine similarity, handling sign ambiguity.

    Args:
        estimated: Shape (K, d) estimated directions.
        true: Shape (K, d) true directions.

    Returns:
        Tuple of (matched errors per component, permutation indices).
    """
    K = estimated.shape[0]
    # Cosine similarity matrix
    sim = torch.abs(estimated @ true.T)  # (K, K)

    matched = []
    used_true = set()
    errors = []

    for _ in range(K):
        # Mask already-used true directions
        sim_masked = sim.clone()
        for j in used_true:
            sim_masked[:, j] = -1.0
        for i in matched:
            sim_masked[i, :] = -1.0

        # Find best match
        idx = sim_masked.argmax()
        i_est = idx // K
        j_true = idx % K
        matched.append(i_est.item())
        used_true.add(j_true.item())

        # Error: ||b_hat - sign * b_true|| where sign aligns them
        b_hat = estimated[i_est]
        b_true = true[j_true]
        sign = torch.sign(b_hat @ b_true)
        error = (b_hat - sign * b_true).norm().item()
        errors.append(error)

    return torch.tensor(errors), list(range(K))


def recover_programs(
    observations: torch.Tensor,
    n_programs: int,
    env_triples: list[tuple[int, int, int]] | None = None,
    tensor_config: TensorConfig | None = None,
    return_intermediate: bool = False,
    use_cross_env_whitening: bool = True,
) -> RecoveryResult:
    """Recover latent program directions from multi-environment observations.

    Full pipeline:
    1. Compute whitening matrices (cross-env or auto-moment based)
    2. Compute whitened third-order cross-moment tensor
    3. Symmetrize the tensor
    4. Run tensor power method on whitened tensor
    5. Unwhiten to recover program directions in original space

    Args:
        observations: Tensor of shape (N, E, d).
        n_programs: Number of programs K to recover.
        env_triples: Specific environment triples. None = auto-select.
        tensor_config: Configuration for numerical parameters.
        return_intermediate: If True, include intermediate tensors in diagnostics.
        use_cross_env_whitening: If True (default), use cross-environment
            moments for noise-free signal subspace identification. Required
            for non-identity environment transforms.

    Returns:
        RecoveryResult with recovered directions and full diagnostics.
    """
    if tensor_config is None:
        tensor_config = TensorConfig()

    start_time = time.time()
    N, E, d = observations.shape
    diagnostics: dict = {}

    # Step 0: Center observations per environment (required for cumulant-based recovery)
    observations = observations - observations.mean(dim=0, keepdim=True)

    # Step 1: Select environment triples
    if env_triples is None:
        env_triples = _select_env_triples(E)
    logger.info("Using %d environment triples for recovery.", len(env_triples))

    # Step 2: Whitening
    # Cross-env whitening needs >= 2 environments (it averages between-env
    # cross-moments). For a single-environment input, fall back to auto-moment
    # whitening with a clear warning instead of a divide-by-zero SVD crash.
    if use_cross_env_whitening and observations.shape[1] < 2:
        logger.warning(
            "Cross-env whitening requires E>=2 environments; got E=%d. "
            "Falling back to auto-moment whitening.",
            observations.shape[1],
        )
        use_cross_env_whitening = False
    if use_cross_env_whitening:
        whitening = compute_cross_env_whitening(
            observations,
            n_programs=n_programs,
            truncation_threshold=tensor_config.truncation_threshold,
        )
        diagnostics["whitening_method"] = "cross_env"
    else:
        whitening = compute_whitening(
            observations,
            truncation_threshold=tensor_config.truncation_threshold,
        )
        diagnostics["whitening_method"] = "auto_moment"
    for e_id, rank in whitening.effective_ranks.items():
        logger.info("Env %d: effective rank %d, kappa=%.2e", e_id, rank, whitening.condition_numbers[e_id])

    # Step 3: Whitened cross-moments
    # Truncate to n_programs (not effective_rank) per Anandkumar theory:
    # the top-K whitened subspace captures the signal, remaining dimensions are noise.
    r = min(n_programs, min(whitening.effective_ranks.values()))
    cross_moment_norms = []

    # Use same-environment triples and average tensors PER environment
    # (each env's whitened tensor is in its own coordinate system, so we
    # can't average across environments; instead pick the best one).
    whitened_tensors = []
    for triple in env_triples:
        T_triple = compute_whitened_cross_moment(
            observations, whitening, triple
        )
        # Truncate to rank r (= n_programs)
        T_triple = T_triple[:r, :r, :r]
        cross_moment_norms.append(T_triple.norm().item())
        whitened_tensors.append(T_triple)

    # Select the tensor with largest norm (best SNR) rather than averaging,
    # since different environments have incompatible whitened coordinate systems.
    best_idx = max(range(len(whitened_tensors)), key=lambda i: cross_moment_norms[i])
    T_avg = whitened_tensors[best_idx]
    # Use the same environment for unwhitening
    ref_env = env_triples[best_idx][0]
    diagnostics["cross_moment_norms"] = cross_moment_norms
    diagnostics["whitened_tensor_norm"] = T_avg.norm().item()
    diagnostics["reference_env"] = ref_env

    # Step 4: Symmetrize
    T_sym, symmetry_error = _symmetrize_tensor(T_avg)
    diagnostics["symmetry_error"] = symmetry_error
    logger.info("Symmetry error: %.2e", symmetry_error)

    if return_intermediate:
        diagnostics["whitened_tensor"] = T_avg.clone()
        diagnostics["symmetrized_tensor"] = T_sym.clone()

    # Step 5: Tensor decomposition
    decomposition = tensor_decomposition(
        T_sym.to(dtype=torch.float64),
        n_components=n_programs,
        n_restarts=tensor_config.n_restarts,
        max_iters=tensor_config.max_power_iters,
        tolerance=tensor_config.power_tolerance,
        reorthogonalize=True,
    )

    # Step 6: Unwhitening
    # The whitening matrix W = U @ diag(1/sqrt(Lambda)) maps from original
    # to whitened space: z = x @ W. The inverse mapping (whitened -> original)
    # requires the pseudo-inverse of W^T: pinv(W^T) = W @ inv(W^T W).
    # Since W^T W = diag(1/Lambda), inv(W^T W) = diag(Lambda), giving:
    # unwhiten = U @ diag(sqrt(Lambda))
    W = whitening.whitening_matrices[ref_env][:, :r].to(dtype=torch.float64)  # (d, r)
    # Compute unwhitening matrix: pinv(W^T)
    WtW = W.T @ W  # (r, r), should be diag(1/Lambda)
    unwhiten = W @ torch.linalg.inv(WtW)  # (d, r), = U @ diag(sqrt(Lambda))
    q_components = decomposition.components.to(dtype=torch.float64)  # (K, r)

    directions = torch.zeros(n_programs, d, dtype=observations.dtype, device=observations.device)
    for k in range(n_programs):
        q_k = q_components[k]  # (r,)
        b_k = unwhiten @ q_k  # (d,)
        b_k = b_k / torch.linalg.norm(b_k)
        directions[k] = b_k.to(dtype=observations.dtype)

    eigenvalues = decomposition.eigenvalues.to(dtype=observations.dtype)

    # Diagnostics
    overlap = torch.abs(directions @ directions.T)
    diagnostics["overlap_matrix"] = overlap
    if n_programs > 1:
        off_diag = overlap - torch.eye(n_programs, device=overlap.device)
        diagnostics["max_off_diagonal_overlap"] = off_diag.abs().max().item()
    else:
        diagnostics["max_off_diagonal_overlap"] = 0.0

    eig_sorted = eigenvalues.abs().sort(descending=True).values
    if len(eig_sorted) > 1:
        diagnostics["spectral_gap"] = (eig_sorted[-1] / eig_sorted[0]).item()
    else:
        diagnostics["spectral_gap"] = 1.0

    diagnostics["recovery_time_seconds"] = time.time() - start_time
    logger.info("Recovery completed in %.2f seconds.", diagnostics["recovery_time_seconds"])

    return RecoveryResult(
        directions=directions,
        eigenvalues=eigenvalues,
        whitening=whitening,
        decomposition=decomposition,
        diagnostics=diagnostics,
    )


def recovery_error(
    estimated: torch.Tensor,
    true: torch.Tensor,
    transform: torch.Tensor | None = None,
) -> float:
    """Compute Hungarian-matched recovery error.

    When environment transforms A_e are non-identity, the pipeline recovers
    directions in the reference environment's observation space (A_e b_k,
    not b_k). Pass the reference environment's transform to compare in
    the correct space.

    Args:
        estimated: Shape (K, d) estimated directions.
        true: Shape (K, d) true program directions (b_k).
        transform: Optional (d, d) environment transform A_{e_ref}.
            If provided, true directions are mapped to observation space
            (A_{e_ref} b_k) before comparison.

    Returns:
        Maximum error across matched components.
    """
    if transform is not None:
        # Map true directions to observation space: A_e @ b_k
        true_obs = (transform @ true.T).T  # (K, d)
        true_obs = true_obs / true_obs.norm(dim=1, keepdim=True)
        true = true_obs
    errors, _ = _hungarian_match(estimated, true)
    return errors.max().item()
