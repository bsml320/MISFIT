#!/usr/bin/env python3
"""Aggregate-only, paired CNN loss comparison within CDR stage and scanner.

Raw input arrays and session identifiers are loaded only in memory. Outputs
contain counts, AUROCs, confidence intervals, and reproducibility metadata.
Requires NumPy and the standard library; no SciPy or scikit-learn dependency.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np


CNN_SHA256 = "f2dd546b85767218cd6e2808317b526b99cab1ea53d71fedd1a82dc6ab1105a6"
TAR_SHA256 = "1335379a2e3e10943762ccda7dfa77fbf7f6f5af8b6029c3e4b8d3dacff6e844"
CLINICAL_SHA256 = "b62e0ef5e683bf1a4d52332e6fc0e34829f8908d8a4980bdfd0ca88420c40762"
CLINICAL_MEMBER = "code_release_misfit_20261005/results/provenance/oasis3_cdr_participant_arrays.npz"
BOOTSTRAP_DEFINITION = (
    "5000 attempted draws by default, NumPy default_rng seed 20261011. "
    "Independently within each CDR-stage x scanner stratum, resample its tested "
    "participants with replacement at the original stratum sample size. "
    "Multinomial participant multiplicities are shared by all three seeds and "
    "both methods. Each seed outcome is pred != label. Original eligibility "
    "(tested n >= 8 and both error outcomes) and equal scanner-seed weights "
    "remain fixed. A stage draw is invalid if any originally eligible "
    "scanner-seed cell loses either outcome; discard the entire stage draw "
    "without reweighting. Each scanner interval uses its own complete valid "
    "draws. Point estimates use the original records. Paired 95% percentile "
    "intervals use NumPy's linear 2.5th/97.5th percentiles on valid draws. "
    "Inference therefore conditions on nondegenerate resamples, observed "
    "scanners, fitted models, and fixed scores; it excludes refit uncertainty."
)


def verify_hash(path, expected):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError("Input SHA-256 mismatch; analysis stopped")
    return actual


def align_clinical(cnn, clinical):
    """Return CDR in CNN order, rejecting ambiguous or inconsistent joins."""
    source = np.asarray(cnn["sessions"])
    target = np.asarray(clinical["sessions"])
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("Session keys must be one-dimensional")
    if len(np.unique(source)) != len(source) or len(np.unique(target)) != len(target):
        raise ValueError("Session keys must be unique in both inputs")
    if set(source) != set(target):
        raise ValueError("Input session-key sets do not match")
    lookup = {key: i for i, key in enumerate(target)}
    order = np.array([lookup[key] for key in source], dtype=int)
    if not np.array_equal(cnn["label"], clinical["label"][order]):
        raise ValueError("Aligned clinical labels disagree")
    scanner = cnn["scanners"][cnn["site"]]
    if not np.array_equal(scanner, clinical["scanner"][order]):
        raise ValueError("Aligned scanner labels disagree")
    tested = np.asarray(cnn["tested"], dtype=bool)
    if not np.all(tested == tested[0]):
        raise ValueError("CNN tested masks disagree across seeds")
    if not np.all(clinical["tested_per_seed"][:, order] == tested[0]):
        raise ValueError("CNN and clinical tested masks disagree")
    cdr = clinical["cdr_raw"][order]
    if not np.all(np.isfinite(cdr)) or np.any(cdr < 0):
        raise ValueError("CDR values must be finite and nonnegative")
    if not np.array_equal(cnn["label"], (cdr > 0).astype(int)):
        raise ValueError("Binary labels disagree with CDR > 0 definition")
    return cdr


def source_fold_scores(folds):
    """Equal finite-fold mean and the first finite fold in saved order."""
    folds = np.asarray(folds, dtype=float)
    if folds.ndim != 3 or np.any(np.isinf(folds)):
        raise ValueError("Source folds must be a finite-or-NaN 3D array")
    finite = np.isfinite(folds)
    counts = finite.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Every record needs a finite source-fold score")
    misfit = np.where(finite, folds, 0).sum(axis=1) / counts
    first = np.argmax(finite, axis=1)
    ordinary = np.take_along_axis(folds, first[:, None, :], axis=1)[:, 0, :]
    return misfit, ordinary


def weighted_auc(error, score, weights=None):
    """Tie-aware weighted U statistic; batched multiplicities avoid expansion.

    A one-class weighted sample is non-estimable and returns NaN, never 0.5.
    A two-dimensional weights array returns one AUROC per row.
    """
    error = np.asarray(error)
    score = np.asarray(score, dtype=float)
    if error.ndim != 1 or score.shape != error.shape or error.size == 0:
        raise ValueError("AUROC needs equally sized nonempty vectors")
    if not np.all(np.isin(error, [0, 1])) or not np.all(np.isfinite(score)):
        raise ValueError("AUROC needs binary outcomes and finite scores")
    w = np.ones(error.size) if weights is None else np.asarray(weights, dtype=float)
    scalar = w.ndim == 1
    w = np.atleast_2d(w)
    if w.shape[1] != error.size or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("Weights must be finite nonnegative participant multiplicities")
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1]
    sorted_w = w[:, order]
    sorted_y = error[order].astype(bool)
    positive = np.add.reduceat(sorted_w * sorted_y, starts, axis=1)
    negative = np.add.reduceat(sorted_w * ~sorted_y, starts, axis=1)
    numerator = np.sum(positive * (np.cumsum(negative, axis=1) - 0.5 * negative), axis=1)
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    auc = np.full(denominator.shape, np.nan)
    np.divide(numerator, denominator, out=auc, where=denominator > 0)
    return float(auc[0]) if scalar else auc


def _summary(points, draws, attempts):
    """Equal original-cell weights; retain only complete finite draw vectors."""
    count = len(points)
    summary = {"eligible_scanner_seed_cells": count,
               "misfit_mean": None, "ordinary_mean": None, "delta_mean": None,
               "misfit_population_sd": None, "ordinary_population_sd": None,
               "delta_population_sd": None}
    boot = {"attempted": attempts if count else 0, "valid": 0, "invalid": 0,
            "misfit_ci95": None, "ordinary_ci95": None, "delta_ci95": None}
    if count:
        original = np.asarray(points, dtype=float)
        paired = np.c_[original, original[:, 0] - original[:, 1]]
        for index, name in enumerate(("misfit", "ordinary", "delta")):
            summary[name + "_mean"] = float(paired[:, index].mean())
            summary[name + "_population_sd"] = float(paired[:, index].std(ddof=0))
        # Each entry is [attempt, method] for an originally eligible cell.
        stacked = np.stack(draws, axis=1)
        valid = np.all(np.isfinite(stacked), axis=(1, 2))
        boot["valid"] = int(valid.sum())
        boot["invalid"] = int(attempts - valid.sum())
        if np.any(valid):
            mean = stacked[valid].mean(axis=1)
            paired_mean = np.c_[mean, mean[:, 0] - mean[:, 1]]
            for index, name in enumerate(("misfit", "ordinary", "delta")):
                boot[name + "_ci95"] = np.percentile(
                    paired_mean[:, index], [2.5, 97.5], method="linear").tolist()
    summary["bootstrap"] = boot
    return summary


def analyze(data, attempts=5000, rng_seed=20261011):
    """Return strictly aggregate results; raw input keys are never serialized."""
    if attempts < 1:
        raise ValueError("At least one bootstrap attempt is required")
    tested = np.asarray(data["tested"], dtype=bool)
    if not np.all(tested == tested[0]):
        raise ValueError("Shared participant bootstrap requires identical tested masks")
    rng = np.random.default_rng(rng_seed)
    result = {
        "analysis": "Within-CDR-stage x scanner CNN epoch-50 loss comparison",
        "primary_stage": "CDR 0.5",
        "eligibility": "Original tested n >= 8 and both held-scanner error outcomes per seed",
        "summary_weighting": "Equal weight for each originally eligible scanner-seed AUROC",
        "sd_definition": "Population SD (ddof=0) across eligible original scanner-seed AUROCs; within one scanner, across eligible seeds",
        "score_definitions": {
            "misfit": "Equal mean of finite epoch-50 source-refit CE within each seed",
            "ordinary": "First finite epoch-50 source-refit CE in saved scanner order",
            "error": "Held-scanner pred != binary label; untested records excluded",
            "direction": "Larger score indicates greater error risk; ties contribute one half",
        },
        "bootstrap": {"definition": BOOTSTRAP_DEFINITION,
                      "attempts_per_stage": attempts, "rng_seed": rng_seed,
                      "rng": "NumPy default_rng / PCG64",
                      "stratum_order": "CDR 0, CDR 0.5, CDR >=1; saved scanner order; eligible strata only"},
        "numpy_version": np.__version__,
        "stages": [],
    }
    cdr = np.asarray(data["cdr"])
    for name, mask, role in (("CDR 0", cdr == 0, "exploratory"),
                             ("CDR 0.5", cdr == 0.5, "primary"),
                             ("CDR >=1", cdr >= 1, "exploratory")):
        stage = {"name": name, "role": role,
                 "error_type": "false positive" if name == "CDR 0" else "false negative",
                 "n_participants": int(mask.sum()),
                 "n_tested_participants": int(np.sum(mask & tested[0])),
                 "cells": []}
        stage_points, stage_draws = [], []
        for scanner_index, scanner in enumerate(data["scanners"]):
            cell_mask = mask & (data["site"] == scanner_index)
            participants = np.flatnonzero(cell_mask & tested[0])
            n = len(participants)
            cell = {"scanner": str(scanner), "n_participants": int(cell_mask.sum()), "seeds": []}
            cell_points, eligible = [], []
            for seed_index, seed in enumerate(data["seeds"]):
                y = data["error"][seed_index, participants]
                errors = int(np.sum(y))
                reason = "n < 8" if n < 8 else ("single outcome" if errors in (0, n) else None)
                seed_result = {"seed": int(seed), "n_tested": n,
                               "n_error": errors, "n_correct": n - errors,
                               "eligible": reason is None, "reason": reason,
                               "misfit_auc": None, "ordinary_auc": None, "delta": None}
                if reason is None:
                    point = [weighted_auc(y, data[method][seed_index, participants])
                             for method in ("misfit", "ordinary")]
                    seed_result.update(misfit_auc=point[0], ordinary_auc=point[1],
                                       delta=point[0] - point[1])
                    cell_points.append(point)
                    eligible.append(seed_index)
                cell["seeds"].append(seed_result)
            cell_draws = []
            if eligible:
                # Counts, outcomes and both scores remain paired across seeds.
                counts = rng.multinomial(n, np.full(n, 1.0 / n), size=attempts)
                for seed_index in eligible:
                    y = data["error"][seed_index, participants]
                    method_draws = np.column_stack([
                        weighted_auc(y, data[method][seed_index, participants], counts)
                        for method in ("misfit", "ordinary")])
                    cell_draws.append(method_draws)
            cell["summary"] = _summary(cell_points, cell_draws, attempts)
            stage_points.extend(cell_points)
            stage_draws.extend(cell_draws)
            stage["cells"].append(cell)
        stage["summary"] = _summary(stage_points, stage_draws, attempts)
        result["stages"].append(stage)
    return result


def load_verified_inputs(cnn_path, release_tar=None, *, clinical_npz=None):
    """Hash validation precedes loading; tar member is never extracted to disk."""
    cnn_hash = verify_hash(cnn_path, CNN_SHA256)
    if (release_tar is None) == (clinical_npz is None):
        raise ValueError("Supply exactly one clinical NPZ or original release archive")
    tar_hash = None
    if clinical_npz is not None:
        verify_hash(clinical_npz, CLINICAL_SHA256)
        payload = Path(clinical_npz).read_bytes()
    else:
        tar_hash = verify_hash(release_tar, TAR_SHA256)
        with tarfile.open(release_tar, "r:gz") as archive:
            stream = archive.extractfile(CLINICAL_MEMBER)
            if stream is None:
                raise ValueError("Clinical provenance member is absent")
            payload = stream.read()
    member_hash = hashlib.sha256(payload).hexdigest()
    if member_hash != CLINICAL_SHA256:
        raise ValueError("Clinical member SHA-256 mismatch; analysis stopped")
    with np.load(cnn_path, allow_pickle=False) as archive:
        cnn = {key: archive[key] for key in ("sessions", "label", "site", "scanners", "seeds", "ce_ep50", "pred", "tested")}
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        clinical = {key: archive[key] for key in ("sessions", "label", "scanner", "cdr_raw", "tested_per_seed")}
    if (cnn["sessions"].shape != (1360,) or cnn["ce_ep50"].shape != (3, 6, 1360)
            or cnn["pred"].shape != (3, 1360) or cnn["tested"].shape != (3, 1360)
            or cnn["label"].shape != (1360,) or cnn["site"].shape != (1360,)
            or cnn["scanners"].shape != (6,) or not np.array_equal(cnn["seeds"], [42, 43, 44])):
        raise ValueError("CNN input schema differs from the approved schema")
    if np.any(cnn["site"] < 0) or np.any(cnn["site"] >= 6):
        raise ValueError("Invalid scanner index")
    if not np.all(np.isin(cnn["pred"][cnn["tested"]], [0, 1])):
        raise ValueError("Tested predictions must be binary")
    if np.isfinite(cnn["ce_ep50"][:, cnn["site"], np.arange(1360)]).any():
        raise ValueError("A target-scanner fold unexpectedly contains source scores")
    # OASIS session naming has a participant key before the first underscore.
    unique_participants = len({str(key).split("_", 1)[0] for key in cnn["sessions"]})
    if unique_participants != 1360:
        raise ValueError("Repeated participant keys require a longitudinal cluster design")
    cdr = align_clinical(cnn, clinical)
    misfit, ordinary = source_fold_scores(cnn["ce_ep50"])
    data = {"scanners": cnn["scanners"], "seeds": cnn["seeds"], "site": cnn["site"],
            "cdr": cdr, "tested": cnn["tested"], "error": cnn["pred"] != cnn["label"],
            "misfit": misfit, "ordinary": ordinary}
    provenance = {
        "cnn_file": Path(cnn_path).name, "cnn_sha256": cnn_hash,
        "clinical_release_file": Path(release_tar).name if release_tar is not None else None,
        "clinical_file": Path(clinical_npz).name if clinical_npz is not None else None, "clinical_release_sha256": tar_hash,
        "clinical_member": CLINICAL_MEMBER, "clinical_member_sha256": member_hash,
        "n_records": 1360, "n_unique_sessions": 1360, "n_unique_participants": unique_participants,
        "n_tested_per_seed": cnn["tested"].sum(axis=1).tolist(),
        "seeds": cnn["seeds"].tolist(), "saved_scanner_order": cnn["scanners"].tolist(),
        "alignment_checks": "Unique complete session-key bijection; binary labels, scanner labels, CDR definition, CNN and clinical tested masks agree",
        "input_handling": "NPZ allow_pickle=False; clinical tar member read in memory; no participant identifiers or raw records written",
    }
    return data, provenance


def render_report(result):
    lines = [result["analysis"], "", "Primary: CDR 0.5; CDR 0 and CDR >=1 are exploratory.",
             result["summary_weighting"], result["sd_definition"], "",
             "Paired participant bootstrap with fixed eligibility and weights:",
             result["bootstrap"]["definition"],
             f"Actual attempted draws per estimable stage: {result['bootstrap']['attempts_per_stage']}; RNG seed: {result['bootstrap']['rng_seed']}", ""]
    def value(x):
        return "NE" if x is None else f"{x:.6f}"
    def ci(x):
        return "NE" if x is None else f"[{x[0]:.6f}, {x[1]:.6f}]"
    def summary_text(summary):
        boot = summary["bootstrap"]
        return (f"MISFIT {value(summary['misfit_mean'])} +/- {value(summary['misfit_population_sd'])}, "
                f"ordinary {value(summary['ordinary_mean'])} +/- {value(summary['ordinary_population_sd'])}, "
                f"delta {value(summary['delta_mean'])} +/- {value(summary['delta_population_sd'])}; "
                f"95% paired percentile CIs MISFIT {ci(boot['misfit_ci95'])}, ordinary {ci(boot['ordinary_ci95'])}, delta {ci(boot['delta_ci95'])}; "
                f"bootstrap valid/attempted {boot['valid']}/{boot['attempted']} (invalid {boot['invalid']})")
    for stage in result["stages"]:
        lines += [f"{stage['name']} ({stage['role']}; {stage['error_type']}): "
                  f"{stage['n_participants']} available, {stage['n_tested_participants']} tested participants",
                  "Stage summary: " + summary_text(stage["summary"])]
        for cell in stage["cells"]:
            lines.append(f"  {cell['scanner']}: available n={cell['n_participants']}")
            for seed in cell["seeds"]:
                lines.append(f"    seed {seed['seed']}: tested n={seed['n_tested']}, error={seed['n_error']}, "
                             f"correct={seed['n_correct']}; MISFIT={value(seed['misfit_auc'])}, "
                             f"ordinary={value(seed['ordinary_auc'])}, delta={value(seed['delta'])}" +
                             (f"; excluded ({seed['reason']})" if seed["reason"] else ""))
            lines.append("    Scanner summary: " + summary_text(cell["summary"]))
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
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results/manuscript_analysis/cdr_scanner_20261011")
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
        summary = stage["summary"]
        print(f"{stage['name']}: MISFIT={summary['misfit_mean']:.10f}, "
              f"ordinary={summary['ordinary_mean']:.10f}, delta={summary['delta_mean']:.10f}; "
              f"valid draws={summary['bootstrap']['valid']}/{summary['bootstrap']['attempted']}")
    print("Wrote aggregate RESULTS.json and ANALYSIS_REPORT.txt")


if __name__ == "__main__":
    main()
