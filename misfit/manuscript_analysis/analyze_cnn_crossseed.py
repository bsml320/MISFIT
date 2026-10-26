#!/usr/bin/env python3
"""Aggregate-only different-score/target-seed CNN comparisons within CDR/scanner.

The three fitted seeds are reused in all six ordered off-diagonal pairs. This
is a diagnostic with different score and target seeds within each comparison;
it neither introduces new fitted models nor creates independent seed sets.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from misfit.manuscript_analysis.analyze_cnn_cdr_scanner import (
    load_verified_inputs as load_same_seed_inputs,
    source_fold_scores,
    verify_hash,
    weighted_auc,
)


METHODS = ("misfit_ce", "firstfinite_ce", "firstfinite_aum", "firstfinite_el2n",
           "crossrefit_aum", "crossrefit_el2n")
METHOD_LABELS = {
    "misfit_ce": "Cross-refit CE (MISFIT)",
    "firstfinite_ce": "First-finite CE",
    "firstfinite_aum": "First-finite saved AUM",
    "firstfinite_el2n": "First-finite epoch-20 EL2N",
    "crossrefit_aum": "Cross-refit saved AUM",
    "crossrefit_el2n": "Cross-refit epoch-20 EL2N",
}
BOOTSTRAP_DEFINITION = (
    "5000 attempted draws by default using NumPy Generator(PCG64(20261011)). "
    "Independently within each CDR-stage x scanner stratum, resample tested "
    "participants with replacement at the original stratum sample size using "
    "multinomial multiplicities. Share the same participant multiplicities "
    "across all six methods, three score seeds, three target outcomes, and six "
    "ordered different-score/target-seed pairs. Original target-cell eligibility "
    "(tested n >= 8 and both outcomes) and equal scanner-pair weights remain "
    "fixed. Reject an entire stage draw if any originally eligible target cell "
    "loses either outcome; do not reweight surviving cells. Scanner summaries "
    "use their own complete valid draws; individual pair intervals use that "
    "pair's valid draws. Report every attempted, valid, and invalid count. "
    "Compute unadjusted pointwise 95% percentile intervals with NumPy linear "
    "2.5th/97.5th percentiles. Intervals condition on nondegenerate resamples, "
    "observed scanners, fitted models, and fixed scores; they exclude refit "
    "uncertainty and do not support confirmatory significance claims."
)
AUM_BOUNDARY = (
    "Use saved AUM without a second negation. The recovered producer "
    "run_oasis3_cnn18_study.py saves the negative mean true-class-minus-other "
    "logit margin over completed epochs 1-50, already oriented toward difficulty."
)


def build_scores(ce_ep50, aum, el2n):
    """Common finite masks guarantee the identical first source refit."""
    folds = [np.asarray(values, dtype=float) for values in (ce_ep50, aum, el2n)]
    if any(values.ndim != 3 or np.any(np.isinf(values)) for values in folds):
        raise ValueError("Source scores must be finite-or-NaN 3D arrays")
    if any(values.shape != folds[0].shape for values in folds[1:]):
        raise ValueError("Source score array shapes disagree")
    mask = np.isfinite(folds[0])
    if any(not np.array_equal(np.isfinite(values), mask) for values in folds[1:]):
        raise ValueError("All methods must have identical finite source-fold masks")
    means, firsts = zip(*(source_fold_scores(values) for values in folds))
    return dict(zip(METHODS, (means[0], firsts[0], firsts[1], firsts[2], means[1], means[2])))


def _validate_data(data, attempts):
    if not isinstance(attempts, (int, np.integer)) or attempts < 1:
        raise ValueError("At least one integer bootstrap attempt is required")
    seeds, scanners = np.asarray(data["seeds"]), np.asarray(data["scanners"])
    if seeds.shape != (3,) or len(np.unique(seeds)) != 3:
        raise ValueError("Exactly three distinct seeds are required")
    if scanners.ndim != 1 or not len(scanners) or len(np.unique(scanners)) != len(scanners):
        raise ValueError("Scanner names must be a nonempty unique vector")
    cdr, site = np.asarray(data["cdr"]), np.asarray(data["site"])
    if (cdr.ndim != 1 or not len(cdr) or not np.all(np.isfinite(cdr))
            or not np.all((cdr == 0) | (cdr == 0.5) | (cdr >= 1))):
        raise ValueError("CDR values must belong to 0, 0.5, or >=1")
    if (site.shape != cdr.shape or not np.issubdtype(site.dtype, np.integer)
            or np.any(site < 0) or np.any(site >= len(scanners))):
        raise ValueError("Invalid scanner indices")
    tested = np.asarray(data["tested"])
    if tested.shape != (3, len(cdr)) or tested.dtype != np.dtype(bool):
        raise ValueError("Tested masks must be a Boolean seed-by-participant array")
    if not np.all(tested == tested[0]):
        raise ValueError("Shared participant bootstrap requires identical tested masks")
    error = np.asarray(data["error"])
    if error.shape != tested.shape or not np.all(np.isin(error[tested], [0, 1])):
        raise ValueError("Tested target errors must be binary seed-by-participant values")
    if set(data["scores"]) != set(METHODS):
        raise ValueError("Exactly the six approved score methods are required")
    for values in data["scores"].values():
        values = np.asarray(values)
        if values.shape != tested.shape or not np.all(np.isfinite(values[tested])):
            raise ValueError("All tested participants need finite scores in every method and seed")


def _summary(points, draws=None, attempts=0):
    """Summarize fixed original-cell weights and complete bootstrap vectors."""
    count = len(points)
    summary = {"eligible_scanner_pair_cells": count,
               "methods": {m: {"mean": None, "population_sd": None} for m in METHODS},
               "deltas": {m: {"mean": None, "population_sd": None} for m in METHODS[1:]}}
    if count:
        original = np.asarray(points, dtype=float)
        delta = original[:, :1] - original[:, 1:]
        for names, values, section in ((METHODS, original, "methods"),
                                       (METHODS[1:], delta, "deltas")):
            for index, method in enumerate(names):
                summary[section][method].update(mean=float(values[:, index].mean()),
                                               population_sd=float(values[:, index].std(ddof=0)))
    if draws is None:
        return summary
    boot = {"attempted": attempts if count else 0, "valid": 0, "invalid": 0}
    for section in ("methods", "deltas"):
        for method in summary[section].values():
            method["ci95"] = None
    if count:
        stacked = np.stack(draws, axis=1)
        valid = np.all(np.isfinite(stacked), axis=(1, 2))
        boot.update(valid=int(valid.sum()), invalid=int(attempts - valid.sum()))
        if np.any(valid):
            means = stacked[valid].mean(axis=1)
            deltas = means[:, :1] - means[:, 1:]
            for names, values, section in ((METHODS, means, "methods"),
                                           (METHODS[1:], deltas, "deltas")):
                for index, method in enumerate(names):
                    summary[section][method]["ci95"] = np.percentile(
                        values[:, index], [2.5, 97.5], method="linear").tolist()
    summary["bootstrap"] = boot
    return summary


def analyze(data, attempts=5000, rng_seed=20261011):
    """Return aggregates only; never serialize participant rows or identifiers."""
    _validate_data(data, attempts)
    tested, cdr = np.asarray(data["tested"]), np.asarray(data["cdr"])
    pairs = [(s, t) for s in range(3) for t in range(3) if s != t]
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    result = {
        "analysis": "Within-CDR-stage x scanner CNN different-score/target-seed diagnostic",
        "primary_stage": "CDR 0.5",
        "method_order": list(METHODS), "method_labels": METHOD_LABELS.copy(),
        "seed_pairs": [{"score_seed": int(data["seeds"][s]), "error_seed": int(data["seeds"][t])}
                       for s, t in pairs],
        "eligibility": "Original target-seed tested n >= 8 and both held-scanner error outcomes",
        "summary_weighting": "Equal weight for each originally eligible scanner x ordered seed-pair AUROC",
        "sd_definition": "Population SD (ddof=0) across eligible original scanner-pair AUROCs or paired deltas; within a scanner, across eligible ordered seed pairs",
        "score_definitions": {
            "misfit_ce": "Equal mean of finite epoch-50 source-refit CE within the score seed",
            "firstfinite_ce": "First finite epoch-50 source-refit CE in saved scanner order within the score seed",
            "firstfinite_aum": "Saved AUM from the same first finite source refit as first-finite CE within the score seed",
            "firstfinite_el2n": "Saved epoch-20 EL2N from the same first finite source refit as first-finite CE within the score seed",
            "crossrefit_aum": "Equal mean of finite saved AUM across source refits within the score seed",
            "crossrefit_el2n": "Equal mean of finite saved epoch-20 EL2N across source refits within the score seed",
            "error": "Different target seed's held-scanner pred != binary label; untested participants excluded",
            "direction": "Larger values indicate greater error risk for all six scores; ties contribute one half",
            "delta": "Cross-refit CE (MISFIT) AUROC minus each comparator AUROC in the identical scanner and ordered seed pair",
            "aum_producer_boundary": AUM_BOUNDARY,
        },
        "bootstrap": {"definition": BOOTSTRAP_DEFINITION, "attempts_per_stage": attempts,
                      "rng_seed": rng_seed, "rng": "NumPy Generator / PCG64",
                      "stratum_order": "CDR 0, CDR 0.5, CDR >=1; saved scanner order; eligible strata only"},
        "interpretation": "All six off-diagonal pairs reuse the same three fitted seeds and are dependent. Different score and target seeds apply within each comparison, not globally disjoint seed sets. Intervals are unadjusted pointwise descriptive diagnostics, with no multiplicity correction or confirmatory superiority claim.",
        "leave_one_scanner_out_definition": "Descriptive point sensitivity: omit one scanner's evaluation cells from the equal-cell mean and population SD. Reuse the existing scores and outcomes; no new training, refitting, or bootstrap inference.",
        "numpy_version": np.__version__, "stages": [],
    }
    for name, mask, role in (("CDR 0", cdr == 0, "exploratory"),
                             ("CDR 0.5", cdr == 0.5, "primary"),
                             ("CDR >=1", cdr >= 1, "exploratory")):
        stage = {"name": name, "role": role,
                 "error_type": "false positive" if name == "CDR 0" else "false negative",
                 "n_participants": int(mask.sum()),
                 "n_tested_participants": int(np.sum(mask & tested[0])), "cells": []}
        stage_points, stage_draws, scanner_points = [], [], []
        for scanner_index, scanner in enumerate(data["scanners"]):
            cell_mask = mask & (np.asarray(data["site"]) == scanner_index)
            index = np.flatnonzero(cell_mask & tested[0])
            n = len(index)
            target_errors = np.asarray(data["error"])[:, index]
            n_error = target_errors.sum(axis=1).astype(int)
            eligible = (n >= 8) & (n_error > 0) & (n_error < n)
            cell = {"scanner": str(scanner), "n_participants": int(cell_mask.sum()),
                    "n_tested_participants": n, "eligible_target_cells": int(eligible.sum()),
                    "seed_pairs": []}
            counts = rng.multinomial(n, np.full(n, 1.0 / n), size=attempts) if np.any(eligible) else None
            cell_points, cell_draws = [], []
            for s, t in pairs:
                reason = "n < 8" if n < 8 else ("single outcome" if not eligible[t] else None)
                pair = {"score_seed": int(data["seeds"][s]), "error_seed": int(data["seeds"][t]),
                        "n_tested": n, "n_error": int(n_error[t]), "n_correct": int(n - n_error[t]),
                        "eligible": reason is None, "reason": reason,
                        "auroc": dict.fromkeys(METHODS), "delta": dict.fromkeys(METHODS[1:])}
                pair_points, pair_draws = [], []
                if reason is None:
                    point = [weighted_auc(target_errors[t], data["scores"][m][s, index]) for m in METHODS]
                    draws = np.column_stack([weighted_auc(target_errors[t], data["scores"][m][s, index], counts)
                                             for m in METHODS])
                    pair["auroc"] = dict(zip(METHODS, point))
                    pair["delta"] = {m: point[0] - point[i] for i, m in enumerate(METHODS) if i}
                    pair_points, pair_draws = [point], [draws]
                    cell_points.append(point)
                    cell_draws.append(draws)
                pair_summary = _summary(pair_points, pair_draws, attempts)
                pair["bootstrap"] = pair_summary["bootstrap"]
                pair["auroc_ci95"] = {m: pair_summary["methods"][m]["ci95"] for m in METHODS}
                pair["delta_ci95"] = {m: pair_summary["deltas"][m]["ci95"] for m in METHODS[1:]}
                cell["seed_pairs"].append(pair)
            cell["summary"] = _summary(cell_points, cell_draws, attempts)
            stage_points.extend(cell_points)
            stage_draws.extend(cell_draws)
            scanner_points.append(cell_points)
            stage["cells"].append(cell)
        stage["eligible_target_cells"] = sum(cell["eligible_target_cells"] for cell in stage["cells"])
        stage["summary"] = _summary(stage_points, stage_draws, attempts)
        stage["leave_one_scanner_out"] = []
        for omitted_index, scanner in enumerate(data["scanners"]):
            retained = [point for index, points in enumerate(scanner_points) if index != omitted_index for point in points]
            sensitivity = {"omitted_scanner": str(scanner), **_summary(retained)}
            stage["leave_one_scanner_out"].append(sensitivity)
        result["stages"].append(stage)
    return result


def load_verified_inputs(cnn_path, release_tar=None, *, clinical_npz=None):
    """Reuse full input hashes, unique-person checks and clinical alignment."""
    data, provenance = load_same_seed_inputs(cnn_path, release_tar, clinical_npz=clinical_npz)
    with np.load(cnn_path, allow_pickle=False) as archive:
        ce, aum, el2n = (archive[key] for key in ("ce_ep50", "aum", "el2n"))
    data["scores"] = build_scores(ce, aum, el2n)
    finite = np.isfinite(ce)
    if not np.all(finite == finite[0]):
        raise ValueError("Finite source masks differ across saved seeds")
    source_counts = finite.sum(axis=1)
    if not np.all(source_counts[data["tested"]] == 4):
        raise ValueError("Every tested participant must have four containing source refits")
    if not np.all(data["tested"].sum(axis=1) == 1358):
        raise ValueError("Expected exactly 1358 tested unique participants per seed")
    # The shared loader already verifies absence in the own-scanner refit;
    # common source masks extend this property to saved AUM and EL2N.
    del data["misfit"], data["ordinary"]
    provenance.update({
        "n_tested_unique_participants": 1358,
        "score_array_shape": list(ce.shape),
        "score_fields": ["ce_ep50", "aum", "el2n"],
        "source_refit_checks": "All three fields have identical finite masks; masks also agree across seeds; each tested participant has four containing refits; own-scanner-excluded refit has no source score; first-finite refit is identical across methods",
        "source_refits_per_tested_participant": 4,
        "shared_analysis_code_file": "scripts/analyze_cnn_cdr_scanner.py",
        "shared_analysis_code_sha256": hashlib.sha256(
            Path(__file__).with_name("analyze_cnn_cdr_scanner.py").read_bytes()).hexdigest(),
    })
    return data, provenance


def render_report(result):
    def number(value):
        return "NE" if value is None else f"{value:.6f}"

    def interval(value):
        return "NE" if value is None else f"[{value[0]:.6f}, {value[1]:.6f}]"

    def summary_lines(summary, prefix):
        lines = [f"{prefix}Eligible scanner-pair cells: {summary['eligible_scanner_pair_cells']}"]
        for section, title in (("methods", "AUROC"), ("deltas", "MISFIT minus")):
            for method, values in summary[section].items():
                line = (f"{prefix}{title} {METHOD_LABELS[method]}: mean {number(values['mean'])}; "
                        f"population SD {number(values['population_sd'])}")
                if "ci95" in values:
                    line += f"; pointwise 95% CI {interval(values['ci95'])}"
                lines.append(line)
        if "bootstrap" in summary:
            boot = summary["bootstrap"]
            lines.append(f"{prefix}Bootstrap valid/attempted {boot['valid']}/{boot['attempted']}; invalid {boot['invalid']}")
        return lines

    lines = [result["analysis"], "", "Primary: CDR 0.5; CDR 0 and CDR >=1 are exploratory.",
             result["eligibility"], result["summary_weighting"], result["sd_definition"], "",
             "Score definitions:"]
    lines.extend(f"  {name}: {definition}" for name, definition in result["score_definitions"].items())
    lines += ["", result["interpretation"], "", "Paired participant bootstrap:",
              result["bootstrap"]["definition"],
              f"Actual attempted draws per estimable stage: {result['bootstrap']['attempts_per_stage']}; RNG seed: {result['bootstrap']['rng_seed']}",
              result["leave_one_scanner_out_definition"], ""]
    for stage in result["stages"]:
        lines += [f"{stage['name']} ({stage['role']}; {stage['error_type']}): "
                  f"{stage['n_participants']} available, {stage['n_tested_participants']} tested participants; "
                  f"{stage['eligible_target_cells']} eligible target cells"]
        lines += summary_lines(stage["summary"], "  ")
        for cell in stage["cells"]:
            lines.append(f"  {cell['scanner']}: available n={cell['n_participants']}, tested n={cell['n_tested_participants']}")
            for pair in cell["seed_pairs"]:
                lines.append(f"    Score seed -> error seed {pair['score_seed']} -> {pair['error_seed']}: "
                             f"n={pair['n_tested']}, error={pair['n_error']}, correct={pair['n_correct']}" +
                             (f"; excluded ({pair['reason']})" if pair["reason"] else ""))
                for method in METHODS:
                    line = f"      {METHOD_LABELS[method]}: AUROC {number(pair['auroc'][method])}; CI {interval(pair['auroc_ci95'][method])}"
                    if method != METHODS[0]:
                        line += f"; MISFIT minus comparator {number(pair['delta'][method])}; CI {interval(pair['delta_ci95'][method])}"
                    lines.append(line)
                boot = pair["bootstrap"]
                lines.append(f"      Bootstrap valid/attempted {boot['valid']}/{boot['attempted']}; invalid {boot['invalid']}")
            lines += summary_lines(cell["summary"], "    ")
        lines.append("  Descriptive leave-one-scanner-out point sensitivity (no new training):")
        for sensitivity in stage["leave_one_scanner_out"]:
            lines.append("    Omitted scanner: " + sensitivity["omitted_scanner"])
            lines += summary_lines(sensitivity, "      ")
        lines.append("")
    if "provenance" in result:
        lines += ["Reproducibility and aggregate-only provenance:", json.dumps(result["provenance"], indent=2)]
    lines.append("NumPy version: " + result["numpy_version"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-path", type=Path, required=True)
    clinical = parser.add_mutually_exclusive_group(required=True)
    clinical.add_argument("--release-tar", type=Path, help="Original October 5 archive")
    clinical.add_argument("--clinical-npz", type=Path, help="Byte-identical clinical NPZ from an authorized archive")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results/manuscript_analysis/crossseed_cdr_scanner_20261011")
    parser.add_argument("--attempts", type=int, default=5000)
    parser.add_argument("--rng-seed", type=int, default=20261011)
    args = parser.parse_args()
    data, provenance = load_verified_inputs(args.cnn_path, args.release_tar, clinical_npz=args.clinical_npz)
    result = analyze(data, attempts=args.attempts, rng_seed=args.rng_seed)
    provenance["analysis_code_file"] = Path(__file__).name
    provenance["analysis_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["provenance"] = provenance
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "RESULTS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "ANALYSIS_REPORT.txt").write_text(render_report(result))
    for stage in result["stages"]:
        summary, boot = stage["summary"], stage["summary"]["bootstrap"]
        print(f"{stage['name']}: MISFIT={summary['methods']['misfit_ce']['mean']}; valid draws={boot['valid']}/{boot['attempted']}")
        for method, delta in summary["deltas"].items():
            print(f"  MISFIT minus {METHOD_LABELS[method]}: {delta['mean']}; CI={delta['ci95']}")
    print("Wrote aggregate RESULTS.json and ANALYSIS_REPORT.txt")


if __name__ == "__main__":
    main()
