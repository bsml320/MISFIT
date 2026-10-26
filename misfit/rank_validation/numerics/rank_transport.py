"""NumPy-only helpers for conditional pair-rank transport checks.

For R_t=(1-t)R_0+tR_1, a smooth minimizer path theta(t), and d=ell_i-ell_j,
Psi(t)=grad(d)^T H_t^{-1} grad(R_1-R_0), evaluated at theta(t).
The exact theorem gives d(theta(1))-d(theta(0))=-integral_0^1 Psi(t) dt.

These helpers evaluate the integrand and approximate its integral. They do not
establish existence, uniqueness, differentiability, or the accuracy of a supplied
minimizer path. In particular, numerical quadrature is not a certified integral.
The examples and tests are synthetic numerical sanity checks, not MRI validation
or verification of assumptions for a trained CNN.
"""

from dataclasses import dataclass
from math import fsum, isfinite, nextafter

import numpy as np


def _finite_scalar(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _round_finite(value, direction, name):
    return _finite_scalar(nextafter(_finite_scalar(value, name), direction), name)


def _vector(value, name, shape=None):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def transport_integrand(t, theta_at, grad_gap, hessian_at, grad_delta_at):
    """Evaluate Psi(t), solving a linear system without forming H^{-1}.

    Callbacks: theta_at(t), grad_gap(theta), hessian_at(t, theta), and
    grad_delta_at(t, theta). The last callback must evaluate grad(R_1-R_0)
    at theta; its t argument is supplied for a uniform path-callback interface.

    This implementation requires a finite symmetric positive definite Hessian.
    It rejects singular or indefinite matrices; it does not add damping or use
    a pseudoinverse. Asymmetry at floating-point noise scale is symmetrized.
    """
    t = _finite_scalar(t, "t")
    if not 0 <= t <= 1:
        raise ValueError("t must lie in [0, 1]")
    theta = _vector(theta_at(t), "theta")
    gradient_gap = _vector(grad_gap(theta), "grad_gap", theta.shape)
    gradient_delta = _vector(grad_delta_at(t, theta), "grad_delta", theta.shape)
    hessian = np.asarray(hessian_at(t, theta), dtype=float)
    if hessian.shape != (theta.size, theta.size) or not np.all(np.isfinite(hessian)):
        raise ValueError("Hessian must be a finite square matrix matching theta")
    if not np.allclose(hessian, hessian.T, rtol=1e-12, atol=1e-14):
        raise ValueError("Hessian must be symmetric")
    hessian = 0.5 * hessian + 0.5 * hessian.T
    try:
        np.linalg.cholesky(hessian)
        solution = np.linalg.solve(hessian, gradient_delta)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Hessian must be positive definite and nonsingular") from exc
    return _finite_scalar(gradient_gap @ solution, "transport integrand")


def first_order_change(theta_at, grad_gap, hessian_at, grad_delta_at):
    """Return -Psi(0); this value alone is not a pair-rank certificate."""
    return -transport_integrand(0.0, theta_at, grad_gap, hessian_at, grad_delta_at)


def integrate_transport(theta_at, grad_gap, hessian_at, grad_delta_at, *, panels=128):
    """Approximate -integral Psi by composite Simpson quadrature on [0, 1].

    panels must be a positive even integer. No quadrature error bound is
    inferred from panel refinement or returned by this numerical approximation.
    """
    if isinstance(panels, (bool, np.bool_)) or not isinstance(panels, (int, np.integer)) or panels < 2 or panels % 2:
        raise ValueError("panels must be a positive even integer")
    values = [transport_integrand(k / panels, theta_at, grad_gap, hessian_at, grad_delta_at)
              for k in range(panels + 1)]
    weighted = [value * (1 if k in (0, panels) else 4 if k % 2 else 2)
                for k, value in enumerate(values)]
    return _finite_scalar(-fsum(weighted) / (3 * panels), "transport integral")


@dataclass(frozen=True)
class PairCertificate:
    """Conditional endpoint interval and strict rank conclusion for one pair."""

    status: str
    lower: float
    upper: float


def certify_pair(initial_gap, predicted_change, remainder_bound, *, initial_gap_error=0.0):
    """Classify a pair using a supplied, independently justified error bound.

    Required premises: the true initial gap lies within initial_gap_error of
    initial_gap, and the true gap change lies within remainder_bound of
    predicted_change. With the default initial_gap_error=0, initial_gap is
    assumed exact. For the first-order transport approximation,
    predicted_change=-Psi(0) and remainder_bound=K/2 when a genuine global bound
    |Psi'(t)|<=K holds on [0,1]. A sampled derivative maximum is insufficient.

    Supply initial-gap uncertainty separately: including it only in the change
    remainder does not protect the initial sign. This helper propagates
    initial_gap_error to the endpoint interval as well as guarding source order.
    Account for numerical prediction error within remainder_bound. This function
    cannot verify these premises. Interval additions round outward; this does
    not certify numerical minimization, linear solves, or quadrature.

    Returns "preserved" or "inverted" only if both the source and endpoint
    intervals have the required strict signs. Ties, intervals touching zero,
    and uncertain source or endpoint signs return "undetermined".
    """
    initial_gap = _finite_scalar(initial_gap, "initial_gap")
    predicted_change = _finite_scalar(predicted_change, "predicted_change")
    remainder_bound = _finite_scalar(remainder_bound, "remainder_bound")
    initial_gap_error = _finite_scalar(initial_gap_error, "initial_gap_error")
    if remainder_bound < 0:
        raise ValueError("remainder_bound must be nonnegative")
    if initial_gap_error < 0:
        raise ValueError("initial_gap_error must be nonnegative")
    source_lower = _round_finite(initial_gap - initial_gap_error, -np.inf, "source lower endpoint")
    source_upper = _round_finite(initial_gap + initial_gap_error, np.inf, "source upper endpoint")
    lower_center = _round_finite(source_lower + predicted_change, -np.inf, "lower center")
    upper_center = _round_finite(source_upper + predicted_change, np.inf, "upper center")
    lower = _round_finite(lower_center - remainder_bound, -np.inf, "interval lower endpoint")
    upper = _round_finite(upper_center + remainder_bound, np.inf, "interval upper endpoint")
    status = "undetermined"
    if (source_lower > 0 and lower > 0) or (source_upper < 0 and upper < 0):
        status = "preserved"
    elif (source_lower > 0 and upper < 0) or (source_upper < 0 and lower > 0):
        status = "inverted"
    return PairCertificate(status, lower, upper)
