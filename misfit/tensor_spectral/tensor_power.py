"""Robust Tensor Power Method with Deflation.

Decomposes a symmetric 3rd-order tensor into rank-1 components via
power iteration with random restarts and deflation.

Reference: Anandkumar et al., "Tensor Decompositions for Learning Latent
Variable Models", JMLR 2014.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import warnings

import torch

logger = logging.getLogger(__name__)


@dataclass
class TensorDecompositionResult:
    """Result of tensor decomposition.

    Attributes:
        components: Recovered orthonormal directions q_k, shape (K, r).
        eigenvalues: Corresponding eigenvalues lambda_k, shape (K,).
        n_iters: Iterations to convergence per component.
        converged: Whether each component converged.
        residual_norm: Frobenius norm of residual tensor after deflation.
    """

    components: torch.Tensor
    eigenvalues: torch.Tensor
    n_iters: list[int]
    converged: list[bool]
    residual_norm: float


def apply_tensor_to_vector(
    tensor: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
) -> torch.Tensor:
    """Compute T(I, v1, v2) = sum_{k,l} T[:,k,l] v1[k] v2[l].

    Args:
        tensor: Shape (r, r, r).
        v1: Shape (r,).
        v2: Shape (r,).

    Returns:
        Shape (r,) result vector.
    """
    return torch.einsum("jkl,k,l->j", tensor, v1, v2)


def _power_iteration_single(
    tensor: torch.Tensor,
    theta: torch.Tensor,
    max_iters: int,
    tolerance: float,
) -> tuple[torch.Tensor, float, int, bool]:
    """Run a single power iteration from a given starting vector.

    Args:
        tensor: Symmetric tensor of shape (r, r, r).
        theta: Initial unit vector of shape (r,).
        max_iters: Maximum number of iterations.
        tolerance: Convergence criterion on ||theta_{t+1} - theta_t||.

    Returns:
        Tuple of (eigenvector, eigenvalue, iterations, converged).
    """
    converged = False
    n_iter = 0
    for t in range(max_iters):
        theta_new = apply_tensor_to_vector(tensor, theta, theta)
        norm = torch.linalg.norm(theta_new)
        if norm < 1e-12:
            # Degenerate direction; return with zero eigenvalue
            n_iter = t + 1
            break
        theta_new = theta_new / norm
        n_iter = t + 1
        if torch.linalg.norm(theta_new - theta) < tolerance:
            converged = True
            theta = theta_new
            break
        theta = theta_new

    eigenvalue = torch.einsum("i,j,k,ijk->", theta, theta, theta, tensor).item()
    return theta, eigenvalue, n_iter, converged


def tensor_power_iteration(
    tensor: torch.Tensor,
    n_restarts: int = 10,
    max_iters: int = 1000,
    tolerance: float = 1e-8,
) -> tuple[torch.Tensor, float, int, bool]:
    """Run tensor power iteration with random restarts to find one component.

    For better coverage, the first restart uses the top singular vector of
    the tensor reshaped as (r, r^2).

    Args:
        tensor: Symmetric tensor of shape (r, r, r).
        n_restarts: Number of random initializations.
        max_iters: Maximum iterations per restart.
        tolerance: Convergence criterion on ||theta_{t+1} - theta_t||.

    Returns:
        Tuple of (eigenvector of shape (r,), eigenvalue, iterations, converged).
    """
    r = tensor.shape[0]
    device = tensor.device
    dtype = tensor.dtype

    best_vec: torch.Tensor | None = None
    best_abs_eigenvalue = -float("inf")
    best_eig_signed = 0.0
    best_iters = 0
    best_converged = False

    for restart in range(n_restarts):
        if restart == 0:
            # Use top singular vector of T reshaped as (r, r^2)
            T_mat = tensor.reshape(r, r * r)
            U, _, _ = torch.linalg.svd(T_mat, full_matrices=False)
            theta = U[:, 0].clone()
        else:
            theta = torch.randn(r, device=device, dtype=dtype)
            theta = theta / torch.linalg.norm(theta)

        vec, eig, iters, conv = _power_iteration_single(
            tensor, theta, max_iters, tolerance
        )

        if abs(eig) > best_abs_eigenvalue:
            best_abs_eigenvalue = abs(eig)
            best_vec = vec
            best_iters = iters
            best_converged = conv
            best_eig_signed = eig

    assert best_vec is not None
    return best_vec, best_eig_signed, best_iters, best_converged


def _gram_schmidt_project(
    vec: torch.Tensor,
    basis: list[torch.Tensor],
) -> torch.Tensor:
    """Project vec onto the orthogonal complement of basis via Gram-Schmidt.

    Args:
        vec: Vector to orthogonalize, shape (r,).
        basis: List of orthonormal vectors.

    Returns:
        Orthogonalized and normalized vector.
    """
    for b in basis:
        vec = vec - torch.dot(vec, b) * b
    norm = torch.linalg.norm(vec)
    if norm < 1e-12:
        warnings.warn(
            "Gram-Schmidt produced near-zero vector; component may be degenerate.",
            stacklevel=2,
        )
        return vec
    return vec / norm


def tensor_decomposition(
    tensor: torch.Tensor,
    n_components: int,
    n_restarts: int = 10,
    max_iters: int = 1000,
    tolerance: float = 1e-8,
    reorthogonalize: bool = True,
) -> TensorDecompositionResult:
    """Decompose symmetric tensor into rank-1 components via power method + deflation.

    Args:
        tensor: Symmetric tensor of shape (r, r, r).
        n_components: Number of components K to recover.
        n_restarts: Random restarts per component.
        max_iters: Max iterations per restart.
        tolerance: Convergence tolerance.
        reorthogonalize: If True, re-orthogonalize recovered components
            after each deflation step to correct accumulating errors.

    Returns:
        TensorDecompositionResult with K components.
    """
    device = tensor.device
    dtype = tensor.dtype

    original_norm = torch.linalg.norm(tensor).item()

    components_list: list[torch.Tensor] = []
    eigenvalues_list: list[float] = []
    n_iters_list: list[int] = []
    converged_list: list[bool] = []

    T_residual = tensor.clone()

    for k in range(n_components):
        vec, eig, iters, conv = tensor_power_iteration(
            T_residual, n_restarts=n_restarts, max_iters=max_iters, tolerance=tolerance
        )

        # Re-orthogonalize against previously recovered components
        if reorthogonalize and components_list:
            vec = _gram_schmidt_project(vec, components_list)
            # Recompute eigenvalue with orthogonalized vector
            eig = torch.einsum(
                "i,j,k,ijk->", vec, vec, vec, T_residual
            ).item()

        components_list.append(vec)
        eigenvalues_list.append(eig)
        n_iters_list.append(iters)
        converged_list.append(conv)

        # Deflate: T <- T - lambda_k * q_k (x) q_k (x) q_k
        T_residual = T_residual - eig * torch.einsum("i,j,k->ijk", vec, vec, vec)

        residual_norm = torch.linalg.norm(T_residual).item()
        if original_norm > 0 and residual_norm / original_norm > 0.5:
            warnings.warn(
                f"After deflating component {k + 1}/{n_components}, "
                f"residual_norm / original_norm = {residual_norm / original_norm:.4f} > 0.5. "
                f"Decomposition may be incomplete.",
                stacklevel=2,
            )

    final_residual_norm = torch.linalg.norm(T_residual).item()

    components = torch.stack(components_list)
    eigenvalues = torch.tensor(eigenvalues_list, device=device, dtype=dtype)

    return TensorDecompositionResult(
        components=components,
        eigenvalues=eigenvalues,
        n_iters=n_iters_list,
        converged=converged_list,
        residual_norm=final_residual_norm,
    )
