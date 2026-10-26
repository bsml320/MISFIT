"""PAC-Bayes generalization bounds for per-subject certificates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class PACBayesConfig:
    """Configuration for PAC-Bayes bound computation."""

    sigma_prior: float = 1.0
    sigma_posterior: float = 0.1
    delta: float = 0.05
    bound_type: str = "kl"  # "kl" or "catoni"


@dataclass
class PACBayesResult:
    """Result of PAC-Bayes bound computation."""

    empirical_losses: torch.Tensor  # (N,) L_hat_i per subject
    kl_divergences: torch.Tensor  # (N,) KL(Q_i || P)
    bounds: torch.Tensor  # (N,) upper bounds on true loss
    sample_counts: torch.Tensor  # (N,) n_i per subject
    is_vacuous: torch.Tensor  # (N,) bool, True if bound > 1


def compute_kl_divergence(
    loadings: torch.Tensor,
    prior_mean: torch.Tensor,
    sigma_prior: float,
    sigma_posterior: float,
) -> torch.Tensor:
    """Compute KL(Q_i || P) for isotropic Gaussians.

    KL(Q_i || P) = (K/2)(sigma_post^2/sigma_prior^2 - 1 + log(sigma_prior^2/sigma_post^2))
                   + (1/(2*sigma_prior^2)) * ||a_i - a_mean||^2

    Args:
        loadings: Subject loadings of shape (N, K).
        prior_mean: Prior mean of shape (K,).
        sigma_prior: Prior standard deviation.
        sigma_posterior: Posterior standard deviation.

    Returns:
        KL divergences of shape (N,).
    """
    K = loadings.shape[1]
    ratio = (sigma_posterior / sigma_prior) ** 2

    # Dimension-dependent term (same for all subjects)
    dim_term = 0.5 * K * (ratio - 1.0 + math.log(1.0 / ratio))

    # Distance-dependent term (varies per subject)
    diff = loadings - prior_mean.unsqueeze(0)
    dist_sq = (diff * diff).sum(dim=1)
    dist_term = dist_sq / (2.0 * sigma_prior**2)

    return dim_term + dist_term


def _kl_bound(
    empirical_losses: torch.Tensor,
    kl_divs: torch.Tensor,
    sample_counts: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Classic KL-Pinsker PAC-Bayes bound.

    L_i <= L_hat_i + sqrt((KL + log(1/delta)) / (2 * n_i))
    """
    log_inv_delta = math.log(1.0 / delta)
    complexity = (kl_divs + log_inv_delta) / (2.0 * sample_counts.clamp(min=1))
    return empirical_losses + complexity.clamp(min=0).sqrt()


def _catoni_bound(
    empirical_losses: torch.Tensor,
    kl_divs: torch.Tensor,
    sample_counts: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Catoni PAC-Bayes bound (tighter for bounded losses in [0, 1]).

    Uses the convex conjugate form:
    L_i <= 1 - exp(-(L_hat_i_catoni + (KL + log(1/delta)) / n_i))
    where L_hat_i_catoni = -log(1 - L_hat_i) (convex transform of empirical loss)
    """
    # Clamp empirical losses to (0, 1) for log safety
    emp_clamped = empirical_losses.clamp(min=1e-8, max=1.0 - 1e-8)
    catoni_emp = -torch.log(1.0 - emp_clamped)

    log_inv_delta = math.log(1.0 / delta)
    complexity = (kl_divs + log_inv_delta) / sample_counts.clamp(min=1)

    return 1.0 - torch.exp(-(catoni_emp + complexity))


def compute_pac_bayes_bounds(
    loadings: torch.Tensor,
    per_subject_losses: torch.Tensor,
    per_subject_counts: torch.Tensor,
    config: PACBayesConfig | None = None,
) -> PACBayesResult:
    """Compute per-subject PAC-Bayes generalization bounds.

    Args:
        loadings: Subject loadings of shape (N, K).
        per_subject_losses: Empirical loss per subject, shape (N,).
        per_subject_counts: Number of samples per subject, shape (N,).
        config: PAC-Bayes configuration.

    Returns:
        PACBayesResult with bounds and diagnostics.
    """
    if config is None:
        config = PACBayesConfig()

    # Data-dependent prior: center on training mean
    prior_mean = loadings.mean(dim=0)

    # KL divergences
    kl_divs = compute_kl_divergence(
        loadings, prior_mean, config.sigma_prior, config.sigma_posterior
    )

    # Compute bounds
    if config.bound_type == "kl":
        bounds = _kl_bound(per_subject_losses, kl_divs, per_subject_counts, config.delta)
    elif config.bound_type == "catoni":
        bounds = _catoni_bound(
            per_subject_losses, kl_divs, per_subject_counts, config.delta
        )
    else:
        raise ValueError(f"Unknown bound_type: {config.bound_type}")

    is_vacuous = bounds > 1.0

    return PACBayesResult(
        empirical_losses=per_subject_losses,
        kl_divergences=kl_divs,
        bounds=bounds,
        sample_counts=per_subject_counts,
        is_vacuous=is_vacuous,
    )
