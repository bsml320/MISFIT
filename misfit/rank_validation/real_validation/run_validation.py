"""Predeclared fixed-feature auxiliary logistic rank-transport validation.

MRI execution requires an audited input and explicit --authorized-aligned-input.
No MRI is loaded by importing this module. Public outputs contain aggregate
statistics only; participant-level arrays are saved exclusively under private/.
Computed residuals are not interval-arithmetic enclosures, so the reported
rank checks are residual-adjusted conditional bound checks, not rigorous
floating-point certificates or certificates for the published CNN.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
from scipy.stats import rankdata


def prepare_features(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Features must be a finite nonempty matrix")
    augmented = np.column_stack((x, np.ones(len(x))))
    scale = float(np.max(np.linalg.norm(augmented, axis=1)))
    return augmented / scale, scale


def eligible_scanners(site, min_held_out=5, min_training=20):
    return [int(s) for s in np.unique(site)
            if np.sum(site == s) >= min_held_out and np.sum(site != s) >= min_training]


def fold_weights(site, excluded):
    retained = np.asarray(site) != excluded
    if not retained.any():
        raise ValueError("An excluded-scanner objective must retain observations")
    return retained.astype(float) / retained.sum()


def logistic_stats(theta, z, y, weights, ridge):
    logits = z @ theta
    probabilities = expit(logits)
    losses = np.logaddexp(0.0, logits) - y * logits
    objective = float(weights @ losses + ridge * (theta @ theta) / 2)
    gradient = z.T @ (weights * (probabilities - y)) + ridge * theta
    hessian = z.T @ ((weights * probabilities * (1 - probabilities))[:, None] * z)
    hessian += ridge * np.eye(z.shape[1])
    return objective, gradient, hessian, losses, probabilities


def fit_logistic(z, y, weights, ridge, *, initial=None, tolerance=1e-10, max_iterations=100):
    if ridge <= 0:
        raise ValueError("Positive ridge penalty is required, including the intercept")
    theta = np.zeros(z.shape[1]) if initial is None else np.array(initial, dtype=float, copy=True)
    for iteration in range(max_iterations + 1):
        objective, gradient, hessian, losses, probabilities = logistic_stats(theta, z, y, weights, ridge)
        if np.linalg.norm(gradient) <= tolerance:
            return {"theta": theta, "gradient": gradient, "hessian": hessian,
                    "objective": objective, "losses": losses, "probabilities": probabilities,
                    "iterations": iteration, "margins": (2 * y - 1) * (z @ theta)}
        direction = cho_solve(cho_factor(hessian, lower=True, check_finite=False), gradient, check_finite=False)
        decrease = float(gradient @ direction)
        step = 1.0
        for _ in range(40):
            trial = theta - step * direction
            trial_logits = z @ trial
            trial_objective = float(weights @ (np.logaddexp(0.0, trial_logits) - y * trial_logits)
                                    + ridge * (trial @ trial) / 2)
            # At the arithmetic floor, permit a full Newton step if the change
            # in objective is smaller than a few ulps; stationarity still gates success.
            allowance = 8 * np.finfo(float).eps * max(1.0, abs(objective))
            if trial_objective <= objective - 1e-4 * step * decrease + allowance:
                theta = trial
                break
            step *= 0.5
        else:
            raise RuntimeError("Newton line search did not converge")
    raise RuntimeError("Newton stationarity tolerance was not attained")


def trace_path(z, y, weights0, weights1, ridge, *, points=65, tolerance=1e-10):
    if points < 3 or points % 2 != 1:
        raise ValueError("An odd number of path points is required")
    theta = np.full(z.shape[1], 0.05)  # independent of zero-initialized endpoint fits
    parameters, gradients, margins, psi, linear_residuals, iterations = [], [], [], [], [], []
    for t in np.linspace(0.0, 1.0, points):
        weights = (1 - t) * weights0 + t * weights1
        fit = fit_logistic(z, y, weights, ridge, initial=theta, tolerance=tolerance)
        theta = fit["theta"]
        delta_gradient = z.T @ ((weights1 - weights0) * (fit["probabilities"] - y))
        direction = cho_solve(cho_factor(fit["hessian"], lower=True, check_finite=False),
                              delta_gradient, check_finite=False)
        parameters.append(theta.copy())
        gradients.append(fit["gradient"].copy())
        margins.append(fit["margins"])
        psi.append((fit["probabilities"] - y) * (z @ direction))
        linear_residuals.append(np.linalg.norm(fit["hessian"] @ direction - delta_gradient))
        iterations.append(fit["iterations"])
    return {"theta": np.asarray(parameters), "gradients": np.asarray(gradients),
            "margins": np.asarray(margins), "psi": np.asarray(psi),
            "linear_residuals": np.asarray(linear_residuals), "iterations": np.asarray(iterations),
            "t": np.linspace(0.0, 1.0, points)}


def integrated_change(psi, points):
    if points < 3 or points % 2 != 1 or (len(psi) - 1) % (points - 1):
        raise ValueError("Quadrature grid must be an odd nested subset of the saved path")
    values = psi[::(len(psi) - 1) // (points - 1)]
    weights = np.ones(points)
    weights[1:-1:2], weights[2:-1:2] = 4, 2
    return -(weights @ values) / (3 * (points - 1))


def pair_constants(z, y, first, second):
    # Globally valid, pair-specific bounds. No fitted outcomes or sampled
    # derivatives are used to choose these constants.
    norms = np.linalg.norm(z, axis=1)
    return norms[first] + norms[second], (norms[first] ** 2 + norms[second] ** 2) / 4


def global_path_bounds(z, weights0, weights1, ridge, c, j):
    norms = np.linalg.norm(z, axis=1)
    weight_change = np.abs(weights1 - weights0)
    delta = float(weight_change @ norms)
    g = float(weight_change @ (norms ** 2) / 4)
    tau = float(max(weights0 @ (norms ** 3), weights1 @ (norms ** 3)) / 4)
    beta = 1 / ridge
    derivative_bound = j * beta ** 2 * delta ** 2 + 2 * c * beta ** 2 * g * delta + c * beta ** 3 * tau * delta ** 2
    lipschitz = j * beta * delta + c * beta * g + c * beta ** 2 * tau * delta
    return {"remainder": derivative_bound / 2, "direct": c * beta * delta,
            "prediction_lipschitz": lipschitz, "delta": delta, "g": g, "tau": tau}


def classify_intervals(initial_gap, predicted_change, change_error, initial_error):
    source_lower = np.nextafter(initial_gap - initial_error, -np.inf)
    source_upper = np.nextafter(initial_gap + initial_error, np.inf)
    lower = np.nextafter(np.nextafter(source_lower + predicted_change, -np.inf) - change_error, -np.inf)
    upper = np.nextafter(np.nextafter(source_upper + predicted_change, np.inf) + change_error, np.inf)
    preserved = ((source_lower > 0) & (lower > 0)) | ((source_upper < 0) & (upper < 0))
    inverted = ((source_lower > 0) & (upper < 0)) | ((source_upper < 0) & (lower > 0))
    return preserved.astype(np.int8) - inverted.astype(np.int8)


def failure_auc(scores, failures):
    failures = np.asarray(failures, dtype=bool)
    positive, negative = int(failures.sum()), int((~failures).sum())
    if not positive or not negative:
        return None
    ranks = rankdata(scores, method="average")
    return float((ranks[failures].sum() - positive * (positive + 1) / 2) / (positive * negative))


def conditional_failure_auc(scores, failures, minimum_count=8):
    if len(scores) == 0:
        return None, "empty"
    if len(scores) < minimum_count:
        return None, "below_minimum_n"
    value = failure_auc(scores, failures)
    return value, "constant_error" if value is None else "defined"


def _groups(y, site, cdr):
    for scanner in np.unique(site):
        for label in (0, 1):
            yield int(scanner), "diagnosis", str(label), np.flatnonzero((site == scanner) & (y == label))
        for stage in np.unique(cdr[np.isfinite(cdr)]):
            yield int(scanner), "cdr", str(float(stage)), np.flatnonzero((site == scanner) & (cdr == stage))
        if np.any((site == scanner) & ~np.isfinite(cdr)):
            yield int(scanner), "cdr", "missing", np.flatnonzero((site == scanner) & ~np.isfinite(cdr))


def _write_csv(path, rows):
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _summary(values):
    return {"mean": float(np.mean(values)), "p95": float(np.quantile(values, 0.95)), "max": float(np.max(values))}


def run_experiment(data, output, *, lambdas=(0.1, 1.0), grids=(17, 33, 65),
                   min_held_out=5, min_training=20, tolerance=1e-10, progress=True, provenance=None):
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() or (output / "settings.json").exists():
        raise FileExistsError("Refusing to overwrite an existing experiment")
    z, scale = prepare_features(data["X"])
    y, site, cdr = np.asarray(data["y"], dtype=float), np.asarray(data["site"], dtype=int), np.asarray(data["cdr_raw"], dtype=float)
    if not (len(y) == len(site) == len(cdr) == len(z)) or not np.isin(y, [0, 1]).all():
        raise ValueError("Aligned binary labels, sites and CDR arrays are required")
    eligible = eligible_scanners(site, min_held_out, min_training)
    if len(eligible) < 2:
        raise ValueError("At least two eligible scanners are required")
    if tuple(grids) != (17, 33, 65):
        raise ValueError("The predeclared quadrature grids are 17, 33 and 65")
    settings = {
        "lambdas": list(lambdas), "quadrature_points": list(grids), "dtype": "float64",
        "stationarity_tolerance_l2": tolerance, "diagnostic_comparison_tolerance": 1e-10,
        "secondary_auc_minimum_count": 8,
        "min_held_out": min_held_out, "min_training": min_training,
        "eligible_scanners": eligible, "represented_scanner_counts": {str(int(s)): int(np.sum(site == s)) for s in np.unique(site)},
        "global_augmented_feature_scale": scale, "augmented_feature_max_norm": float(np.max(np.linalg.norm(z, axis=1))),
        "feature_shape": list(z.shape), "normalization": "One common scale for [MRI features,1]; constant penalized intercept; fixed along all paths",
        "risk": "Unweighted normalized empirical logistic risk excluding one scanner, plus lambda*||theta||^2/2 on every coefficient",
        "quadrature": "Composite Simpson; nested 17/33/65 grids extracted from the same independently optimized 65-point path",
        "initialization": "Independent endpoint fits start at zero; path t=0 starts at all-0.05 vector; later path points warm-start preceding point",
        "bound_constants": "c=||zi||+||zj||; j=(||zi||^2+||zj||^2)/4; delta=sum|a|norm; g=sum|a|norm^2/4; tau=max_endpoint(sum w norm^3)/4",
        "scope": "Fixed-representation auxiliary logistic mechanism check, not published CNN validation or a prospective generalization evaluation",
        "provenance": provenance or {"fixture": "synthetic"},
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
    private = output / "private"
    private.mkdir(exist_ok=True)
    np.savez_compressed(private / "fixed_representation.npz", z=z, y=y, site=site, cdr_raw=cdr, scale=scale)
    rows, path_rows = [], []
    total_paths = 0
    max_stationarity = 0.0
    max_endpoint_distance = 0.0
    weights = {scanner: fold_weights(site, scanner) for scanner in eligible}
    for ridge in lambdas:
        lam_private = private / ("lambda_" + str(ridge).replace(".", "p"))
        lam_private.mkdir()
        endpoints = {scanner: fit_logistic(z, y, weights[scanner], ridge, tolerance=tolerance) for scanner in eligible}
        np.savez_compressed(lam_private / "endpoint_fits.npz", excluded_scanner=np.asarray(eligible),
                            theta=np.asarray([endpoints[s]["theta"] for s in eligible]),
                            gradients=np.asarray([endpoints[s]["gradient"] for s in eligible]),
                            margins=np.asarray([endpoints[s]["margins"] for s in eligible]),
                            losses=np.asarray([endpoints[s]["losses"] for s in eligible]))
        for endpoint in endpoints.values():
            max_stationarity = max(max_stationarity, float(np.linalg.norm(endpoint["gradient"])))
        for target in eligible:
            sources = [source for source in eligible if source != target]
            paths = {}
            for source in sources:
                path = trace_path(z, y, weights[source], weights[target], ridge, points=max(grids), tolerance=tolerance)
                paths[source] = path
                total_paths += 1
                np.savez_compressed(lam_private / f"path_source_{source}_target_{target}.npz", **path)
                residual = float(np.max(np.linalg.norm(path["gradients"], axis=1)))
                endpoint_distance = max(float(np.linalg.norm(path["theta"][0] - endpoints[source]["theta"])),
                                        float(np.linalg.norm(path["theta"][-1] - endpoints[target]["theta"])))
                max_stationarity = max(max_stationarity, residual)
                max_endpoint_distance = max(max_endpoint_distance, endpoint_distance)
                target_idx = np.flatnonzero(site == target)
                for grid in grids:
                    residual_by_subject = (integrated_change(path["psi"], grid)
                                           - (endpoints[target]["losses"] - endpoints[source]["losses"]))
                    path_rows.append({"lambda": ridge, "source_excluded_scanner": source, "target_scanner": target,
                                      "quadrature_points": grid, "max_stationarity_l2": residual,
                                      "independent_endpoint_parameter_distance": endpoint_distance,
                                      "max_pair_identity_error_within_target_scanner": float(np.ptp(residual_by_subject[target_idx]))})
                if progress:
                    print(f"lambda={ridge:g} source={source} target={target}: path solved; residual={residual:.3e}", flush=True)
            source_losses = np.mean([endpoints[s]["losses"] for s in sources], axis=0)
            target_losses = endpoints[target]["losses"]
            prediction = np.mean([-paths[s]["psi"][0] for s in sources], axis=0)
            integrated = {grid: np.mean([integrated_change(paths[s]["psi"], grid) for s in sources], axis=0) for grid in grids}
            target_predictions = (z @ endpoints[target]["theta"] >= 0).astype(int)
            target_failures = target_predictions != y
            for scanner, kind, value, idx in _groups(y, site, cdr):
                if scanner != target:
                    continue
                auc_value, auc_status = conditional_failure_auc(source_losses[idx], target_failures[idx])
                record = {"lambda": ridge, "scanner": scanner, "group_kind": kind, "group_value": value,
                          "participants": len(idx), "pairs": len(idx) * (len(idx) - 1) // 2,
                          "source_folds": len(sources), "target_failures": int(target_failures[idx].sum()),
                          "failure_auc": auc_value, "auc_status": auc_status,
                          "eligibility": "eligible"}
                if len(idx) < 2:
                    record["pair_status"] = "empty_or_singleton"
                    rows.append(record)
                    continue
                a, b = np.triu_indices(len(idx), 1)
                first, second = idx[a], idx[b]
                c, j = pair_constants(z, y, first, second)
                initial = source_losses[first] - source_losses[second]
                final = target_losses[first] - target_losses[second]
                correction = prediction[first] - prediction[second]
                all_bounds = [global_path_bounds(z, weights[s], weights[target], ridge, c, j) for s in sources]
                remainder = np.mean([bound["remainder"] for bound in all_bounds], axis=0)
                direct = np.mean([bound["direct"] for bound in all_bounds], axis=0)
                source_error = c * np.mean([np.linalg.norm(endpoints[s]["gradient"]) / ridge for s in sources])
                target_error = c * np.linalg.norm(endpoints[target]["gradient"]) / ridge
                prediction_error = np.mean([
                    bound["prediction_lipschitz"] * np.linalg.norm(paths[s]["gradients"][0]) / ridge
                    + c * paths[s]["linear_residuals"][0] / ridge
                    for s, bound in zip(sources, all_bounds)], axis=0)
                first_order_error = np.abs(final - initial - correction)
                adjusted_bound = remainder + source_error + target_error + prediction_error
                first_status = classify_intervals(initial, correction, remainder + prediction_error, source_error)
                direct_status = classify_intervals(initial, np.zeros_like(initial), direct, source_error)
                conflict = (first_status == -1) & (direct_status == 1)
                if conflict.any():
                    raise ArithmeticError("Analytical rank bounds disagree")
                combined = np.where(first_status != 0, first_status, direct_status)
                resolved = (np.abs(initial) > source_error + 1e-10) & (np.abs(final) > target_error + 1e-10)
                observed_status = np.where(initial * final > 0, 1, np.where(initial * final < 0, -1, 0))
                false = (combined != 0) & resolved & (combined != observed_status)
                record.update({"pair_status": "evaluated", "initial_exact_float_ties": int(np.sum(initial == 0)),
                               "target_exact_float_ties": int(np.sum(final == 0)),
                               "first_order_bound_max_violation": float(np.max(first_order_error - adjusted_bound)),
                               "first_order_bound_violation_count": int(np.sum(first_order_error > adjusted_bound + 1e-10)),
                               "first_order_rank_checks": int(np.count_nonzero(first_status)),
                               "direct_preservation_checks": int(np.sum(direct_status == 1)),
                               "combined_preservation_checks": int(np.sum(combined == 1)),
                               "combined_inversion_checks": int(np.sum(combined == -1)),
                               "combined_bound_check_fraction": float(np.mean(combined != 0)),
                               "observed_false_bound_checks": int(false.sum()),
                               "bound_checks_with_unresolved_endpoint_sign": int(np.sum((combined != 0) & ~resolved)),
                               "observed_preserved_fraction_resolved": float(np.mean(observed_status[resolved] == 1)) if resolved.any() else None,
                               "numerically_resolved_pair_count": int(resolved.sum()),
                               "initial_gap_error_bound_max": float(np.max(source_error)),
                               "target_gap_error_bound_max": float(np.max(target_error)),
                               "prediction_error_bound_max": float(np.max(prediction_error))})
                for name, values in (("first_order_error", first_order_error), ("analytical_remainder_bound", remainder),
                                     ("absolute_initial_gap", np.abs(initial)), ("direct_change_bound", direct)):
                    record.update({name + "_" + stat: result for stat, result in _summary(values).items()})
                for grid in grids:
                    correction_at_grid = integrated[grid][first] - integrated[grid][second]
                    error = np.abs((final - initial) - correction_at_grid)
                    record.update({f"identity_error_{grid}_{stat}": result for stat, result in _summary(error).items()})
                rows.append(record)
        for scanner, kind, value, idx in _groups(y, site, cdr):
            if scanner not in eligible:
                rows.append({"lambda": ridge, "scanner": scanner, "group_kind": kind, "group_value": value,
                             "participants": len(idx), "pairs": len(idx) * (len(idx) - 1) // 2,
                             "eligibility": "ineligible_scanner", "pair_status": "not_evaluated"})
    _write_csv(output / "group_results.csv", rows)
    _write_csv(output / "path_results.csv", path_rows)
    evaluated = [row for row in rows if row.get("pair_status") == "evaluated"]
    diagnosis = [row for row in evaluated if row["group_kind"] == "diagnosis"]
    aggregates = []
    for ridge in lambdas:
        for kind, stage in (("diagnosis", None), ("cdr", "0.5")):
            selected = [row for row in evaluated if row["lambda"] == ridge and row["group_kind"] == kind
                        and (stage is None or row["group_value"] == stage)]
            denominator = sum(row["pairs"] for row in selected)
            aggregates.append({"lambda": ridge, "group_scope": "diagnosis_by_scanner" if stage is None else "CDR_0.5_by_scanner",
                               "groups": len(selected), "pairs": denominator,
                               "preservation_bound_checks": sum(row["combined_preservation_checks"] for row in selected),
                               "inversion_bound_checks": sum(row["combined_inversion_checks"] for row in selected),
                               "bound_check_fraction": sum(row["combined_preservation_checks"] + row["combined_inversion_checks"] for row in selected) / denominator if denominator else None,
                               "observed_false_bound_checks": sum(row["observed_false_bound_checks"] for row in selected),
                               "first_order_bound_violation_count": sum(row["first_order_bound_violation_count"] for row in selected),
                               **{f"maximum_identity_error_{grid}": max((row[f"identity_error_{grid}_max"] for row in selected), default=None) for grid in grids}})
    summary = {"settings": settings, "path_count": total_paths, "aggregate_results": aggregates,
               "observed_false_bound_checks": sum(row["observed_false_bound_checks"] for row in evaluated),
               "first_order_bound_violation_count": sum(row["first_order_bound_violation_count"] for row in evaluated),
               "maximum_stationarity_l2": max_stationarity,
               "maximum_independent_endpoint_parameter_distance": max_endpoint_distance,
               "group_rows": len(rows), "evaluated_group_rows": len(evaluated),
               "runtime_seconds": time.perf_counter() - start,
               "private_array_bytes": sum(path.stat().st_size for path in private.rglob("*.npz")),
               "runtime": {"python": platform.python_version(), "numpy": np.__version__},
               "interpretation": "Residual-adjusted conditional bound checks; not rigorous floating-point certificates. Stationarity and linear-solve residuals are numerically evaluated, not validated interval bounds. No CNN certificate is inferred.",
               "error_accounting": "Source/target pair loss error <= c*||objective gradient||/lambda. First-order source error <= (j*beta*delta+c*beta*g+c*beta^2*tau*delta)*||source gradient||/lambda, plus c*beta*linear-solve residual. Source sign uncertainty is separately guarded. Quadrature refinement is diagnostic only.",
               "reporting": "All predeclared lambdas, grids, eligible folds, diagnosis/scanner and CDR/scanner groups are retained. Empty, singleton, constant-error and ineligible groups remain in CSV. CDR and diagnosis aggregate sets overlap and are not independent replications."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorized-aligned-input", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--cnn-path", type=Path, required=True)
    parser.add_argument("--clinical-npz", type=Path, required=True)
    args = parser.parse_args()
    if not args.authorized_aligned_input:
        parser.error("MRI execution requires audited alignment and explicit authorization")
    from misfit.rank_validation.real_validation.audit_input import load_inputs
    loaded = load_inputs(args.features, args.cnn_path, args.clinical_npz)
    numeric_data = {key: loaded[key] for key in ("X", "y", "site", "cdr_raw")}
    del loaded
    provenance = {"input_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
                  "alignment": "Auditor PASS: earliest participant session, exact canonical row/label/scanner/CDR alignment",
                  "representation": "Provided 50 MRI features from global voxel mask/z-score/PCA over all 2720 sessions before 1360-participant selection; this is a fixed-representation mechanism check, not fresh prospective generalization."}
    result = run_experiment(numeric_data, args.output, provenance=provenance)
    print(json.dumps({key: result[key] for key in ("path_count", "runtime_seconds", "private_array_bytes", "aggregate_results")}, indent=2))
