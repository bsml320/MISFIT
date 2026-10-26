"""Reproduce quantitative diagnostics from the deterministic test fixtures.

This script reports floating-point measurements. Run the unittest suite
separately for pass/fail assertions. It does not manufacture global error bounds
from the observed diagnostics or from sampled derivatives.
"""

from dataclasses import asdict
import json
import platform

import numpy as np

from misfit.rank_validation.numerics.rank_transport import certify_pair, first_order_change, integrate_transport, transport_integrand
from misfit.rank_validation.numerics.test_rank_transport import SyntheticChecks


def build_report():
    """Return JSON-compatible numerical measurements; no output files are edited."""
    theta, dg, h, dr, exact = SyntheticChecks.matrix_quadratic()
    transported = integrate_transport(theta, dg, h, dr, panels=256)
    matrix_report = {
        "panels": 256,
        "analytic_endpoint_change": float(exact),
        "transport_integral_approximation": transported,
        "absolute_identity_error": float(abs(transported - exact)),
    }

    theta, dg, h, dr, exact, k_bound = SyntheticChecks.scalar_quadratic()
    first_order = first_order_change(theta, dg, h, dr)
    convergence = [
        {"panels": n, "absolute_error": abs(integrate_transport(theta, dg, h, dr, panels=n) - exact)}
        for n in (4, 8, 16, 32, 64)
    ]
    scalar_report = {
        "analytic_endpoint_change": exact,
        "first_order_change": first_order,
        "absolute_first_order_remainder": abs(exact - first_order),
        "analytic_global_derivative_bound": k_bound,
        "analytic_remainder_bound": k_bound / 2,
        "bound_provenance": "Psi(t)=a*C/h(t)^2; K=2*abs(a*C*(h1-h0))/min(h0,h1)^3; no sampled derivative maximum",
        "quadrature_convergence": convergence,
        "preserved_pair": asdict(certify_pair(2.5, first_order, k_bound / 2)),
        "inverted_pair": asdict(certify_pair(-0.3, first_order, k_bound / 2)),
        "uncertain_initial_sign_pair": asdict(certify_pair(1e-16, 1.0, 0.0, initial_gap_error=2e-16)),
    }

    predictions, bounds, initial, final, transports = [], [], [], [], []
    for h0, h1, b0, b1 in SyntheticChecks.containing_fold_parameters():
        theta, dg, h, dr, exact, k_bound = SyntheticChecks.scalar_quadratic(h0, h1, b0, b1)
        predictions.append(first_order_change(theta, dg, h, dr))
        bounds.append(k_bound / 2)
        initial.append(b0 / h0 - 0.8)
        final.append(b1 / h1 - 0.8)
        transports.append(integrate_transport(theta, dg, h, dr, panels=256))
    fold_certificate = certify_pair(np.mean(initial), np.mean(predictions), np.mean(bounds))
    fold_report = {
        "fold_count": len(initial),
        "shared_target_objective": {"h1": 3.0, "b1": 4.0, "minimizer": 4 / 3},
        "initial_average_gap": float(np.mean(initial)),
        "final_average_gap": float(np.mean(final)),
        "absolute_average_identity_error": float(abs(np.mean(transports) - (np.mean(final) - np.mean(initial)))),
        "first_order_average_change": float(np.mean(predictions)),
        "analytic_average_remainder_bound": float(np.mean(bounds)),
        "certificate": fold_certificate.status,
        "endpoint_interval": [fold_certificate.lower, fold_certificate.upper],
    }

    theta, dg, h, dr, gap, objective, ridge, derivative_terms = SyntheticChecks.ridge_logistic()
    exact = float(gap(theta(1)) - gap(theta(0)))
    transported = integrate_transport(theta, dg, h, dr, panels=64)
    derivative_t = 0.37
    terms = derivative_terms(derivative_t)
    component_names = ("gap_curvature", "objective_shift_curvature", "objective_third_derivative")
    analytic_derivative = sum(terms[name] for name in component_names)
    derivative_differences = []
    for step in (2e-3, 1e-3):
        finite_difference = (transport_integrand(derivative_t + step, theta, dg, h, dr)
                             - transport_integrand(derivative_t - step, theta, dg, h, dr)) / (2 * step)
        derivative_differences.append({
            "step": step,
            "central_difference": finite_difference,
            "absolute_error": abs(finite_difference - analytic_derivative),
        })
    logistic_report = {
        "ridge": ridge,
        "panels": 64,
        "independently_evaluated_endpoint_change": exact,
        "transport_integral_approximation": transported,
        "absolute_identity_error": abs(transported - exact),
        "maximum_stationarity_residual_at_checkpoints": float(max(
            np.linalg.norm(objective(t, theta(t))[1], ord=np.inf) for t in (0.0, 0.2, 0.7, 1.0))),
        "transport_derivative": {
            "interior_t": derivative_t,
            "formula": "Psi'=-v^T J v - 2 c^T H^-1 G v + c^T H^-1 T[v] v",
            "terms": terms,
            "analytic_derivative": analytic_derivative,
            "central_differences": derivative_differences,
            "scope": "Pointwise formula sanity check only; not a global derivative bound",
        },
        "remainder_certificate": "Not claimed: no global derivative bound established for this example",
    }
    return {
        "scope": "Synthetic numerical sanity checks only; not MRI validation; not CNN validation",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy_required": False},
        "matrix_quadratic": matrix_report,
        "scalar_quadratic": scalar_report,
        "shared_containing_folds": fold_report,
        "ridge_logistic": logistic_report,
        "interpretation": "Small identity errors support code sanity on these examples; they are not a theorem proof or a verified numerical error bound.",
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, allow_nan=False))
