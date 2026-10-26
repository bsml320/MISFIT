#!/usr/bin/env python3
"""Post hoc CDR analysis; read external records and export aggregate JSON only.

No training, participant exports, row diagnostics or bootstrap indices are
written. The restricted source is supplied only through --input. Synthetic
tests require no source data. See --help for the reproducible run parameters.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44, 45, 46)
LOSS_COLUMNS = [f"misfit_seed{s}" for s in SEEDS]
ERROR_COLUMNS = [f"err_seed{s}" for s in SEEDS]
STAGE = "cdr_stage_0_05_1plus"
STAGE_LABELS = {0: "0", 1: "0.5", 2: "≥1"}
COLUMNS = ["session", "scanner", "scanner_code", "label_impaired1", "cdr_raw",
           STAGE, "age", "sex_male1", "misfit_mean", "misfit_percentile",
           *LOSS_COLUMNS, *ERROR_COLUMNS]
METHOD_SOURCE = ROOT / "misfit/experiments/run_clinical_stratified_media.py"
DEFAULT_OUTPUT = ROOT / "results/manuscript_analysis/cdr_stage_analysis.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_distinct_output(input_path, output_path):
    """Reject direct, symlink and hardlink aliases before reading the source."""
    source, destination = Path(input_path), Path(output_path)
    if (source.resolve() == destination.resolve()
            or (destination.exists() and destination.samefile(source))):
        raise ValueError("Input and output must be different files")


def fixed_percentile(values):
    """Published convention: minimum rank is 1, average tied ranks, divisor N."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Percentile input must be a nonempty finite vector")
    return 100 * rankdata(values, method="average") / len(values)


def load_records(path):
    """Accept the original records or the later release with explicit targets."""
    frame = pd.read_csv(path, float_precision="round_trip", dtype={"session": str, "scanner": str})
    extra = [f"pred_seed{s}" for s in SEEDS] + [f"tested_seed{s}" for s in SEEDS]
    if set(frame.columns) == set(COLUMNS + extra) and len(frame.columns) == 30:
        eligible = tested_mask(frame.scanner_code.to_numpy())
        labels = frame.label_impaired1.to_numpy()
        for seed in SEEDS:
            tested = frame[f"tested_seed{seed}"].to_numpy()
            pred = frame[f"pred_seed{seed}"].to_numpy()
            error = frame[f"err_seed{seed}"].to_numpy()
            if not np.all(np.isin(tested, [0, 1])) or not np.array_equal(tested, eligible):
                raise ValueError("Released tested flags disagree with fold eligibility")
            if not np.all(np.isin(pred[eligible], [0, 1])):
                raise ValueError("Released tested predictions must be binary")
            if not np.array_equal(error[eligible], (pred[eligible] != labels[eligible]).astype(int)):
                raise ValueError("Released predictions and errors disagree")
            if not np.all(pred[~eligible] == -1):
                raise ValueError("Untested predictions must retain the sentinel")
        return frame.loc[:, COLUMNS].copy()
    return frame


def tested_mask(scanner_codes):
    """Mirror released collect(): test >=5 and complementary training >=20."""
    codes = np.asarray(scanner_codes)
    mask = np.zeros(len(codes), dtype=bool)
    for code in np.unique(codes):
        held = codes == code
        if held.sum() >= 5 and (~held).sum() >= 20:
            mask[held] = True
    return mask


def validate_records(frame, expected_n=None, expected_counts=None):
    """Fail with schema-level messages only; never echo offending records."""
    if set(frame.columns) != set(COLUMNS) or len(frame.columns) != 20:
        raise ValueError("Input must contain the expected 20-column schema")
    if frame.empty or frame.isna().any().any():
        raise ValueError("Input is empty or has missing values")
    if expected_n is not None and len(frame) != expected_n:
        raise ValueError("Unexpected cohort size")
    if not frame.session.map(lambda value: isinstance(value, str) and "_MR_" in value).all():
        raise ValueError("Sessions must encode a participant prefix before _MR_")
    participants = frame.session.str.split("_MR_", n=1).str[0]
    if frame.session.nunique() != len(frame) or participants.nunique() != len(frame):
        raise ValueError("Sessions and participant prefixes must be unique")
    try:
        numeric = frame[[c for c in COLUMNS if c not in ("session", "scanner")]].to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Expected numeric fields are not numeric") from None
    if not np.isfinite(numeric).all():
        raise ValueError("Numeric fields must be finite")
    for column in ("label_impaired1", "sex_male1", *ERROR_COLUMNS):
        if not frame[column].isin([0, 1]).all():
            raise ValueError("Binary field contains invalid values")
    if not frame[STAGE].isin([0, 1, 2]).all():
        raise ValueError("Invalid stage code")
    if not np.array_equal(frame.label_impaired1.to_numpy(), (frame[STAGE].to_numpy() > 0).astype(int)):
        raise ValueError("Impaired label disagrees with stage")
    raw = frame.cdr_raw.to_numpy()
    if not np.isin(raw, [0, 0.5, 1, 2, 3]).all():
        raise ValueError("Raw CDR must be a valid recorded stage")
    mapped = np.where(raw == 0, 0, np.where(raw == 0.5, 1, 2))
    if not np.array_equal(frame[STAGE].to_numpy(), mapped):
        raise ValueError("Raw CDR and grouped stage disagree")
    codes = frame.scanner_code.to_numpy()
    if not np.equal(codes, np.floor(codes)).all() or (codes < 0).any():
        raise ValueError("Scanner codes must be nonnegative integers")
    if not frame.scanner.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Scanner labels must be nonempty strings")
    if (frame.groupby("scanner_code").scanner.nunique().max() != 1
            or frame.groupby("scanner").scanner_code.nunique().max() != 1):
        raise ValueError("Scanner label and code mapping is not bijective")
    if (frame[LOSS_COLUMNS].to_numpy() < 0).any():
        raise ValueError("Cross-entropy losses must be nonnegative")
    if not np.array_equal(frame[LOSS_COLUMNS].to_numpy().mean(axis=1), frame.misfit_mean.to_numpy()):
        raise ValueError("Five-seed mean does not exactly reproduce supplied mean")
    if not np.array_equal(fixed_percentile(frame.misfit_mean), frame.misfit_percentile.to_numpy()):
        raise ValueError("Average-rank percentile does not exactly reproduce supplied percentile")
    counts = [int((frame[STAGE] == stage).sum()) for stage in range(3)]
    if any(count == 0 for count in counts):
        raise ValueError("All three CDR groups must be present")
    if expected_counts is not None and counts != list(expected_counts):
        raise ValueError("Unexpected stage counts")
    tested = tested_mask(codes)
    if not (frame.loc[~tested, ERROR_COLUMNS].to_numpy() == 1).all():
        raise ValueError("Untested error entries disagree with canonical sentinel semantics")
    return {
        "n_rows": len(frame), "n_columns": len(frame.columns),
        "n_sessions": int(frame.session.nunique()), "n_participants": int(participants.nunique()),
        "participant_key": "Session prefix before _MR_; uniqueness checked in memory only",
        "stage_counts": counts, "n_scanners": int(frame.scanner_code.nunique()),
        "n_tested": int(tested.sum()), "n_untested": int((~tested).sum()),
        "n_impaired": int(frame.label_impaired1.sum()),
        "n_impaired_tested": int((tested & (frame.label_impaired1.to_numpy() == 1)).sum()),
        "complete_finite_inputs": True, "binary_sex_labels_errors": True,
        "raw_cdr_stage_label_consistent": True, "scanner_label_code_bijection": True,
        "five_seed_mean_exact": True, "percentile_exact": True,
        "csv_float_precision": "round_trip", "seeds": list(SEEDS),
        "untested_error_sentinel_excluded": True,
        "eligibility_rule": "held-scanner test n >= 5 and complementary training n >= 20",
    }


def design_matrix(frame, scanner_levels=None, age_center=None):
    levels = sorted(frame.scanner_code.unique()) if scanner_levels is None else scanner_levels
    center = float(frame.age.mean()) if age_center is None else float(age_center)
    matrix = np.column_stack([
        np.ones(len(frame)), (frame[STAGE].to_numpy() == 1).astype(float),
        frame.age.to_numpy() - center, frame.sex_male1.to_numpy(),
        *[(frame.scanner_code.to_numpy() == code).astype(float) for code in levels[1:]],
    ])
    names = ["intercept", "stage05", "age_centered", "sex_male1",
             *[f"scanner_code_{int(code)}" for code in levels[1:]]]
    return matrix, names, levels, center


def fit_adjusted(frame, y=None):
    if not frame[STAGE].isin([1, 2]).all():
        raise ValueError("Adjusted model is defined only among impaired participants")
    matrix, names, levels, center = design_matrix(frame)
    outcome = frame.misfit_percentile.to_numpy() if y is None else np.asarray(y, dtype=float)
    if not np.isfinite(outcome).all():
        raise ValueError("OLS outcome must be finite")
    coefficients, _, rank, singular_values = np.linalg.lstsq(matrix, outcome, rcond=None)
    estimable = rank == matrix.shape[1]
    reference = str(frame.loc[frame.scanner_code == levels[0], "scanner"].iloc[0])
    return {
        "contrast": "CDR 0.5 minus CDR ≥1", "estimate": float(coefficients[1]) if estimable else None,
        "unit": "cohort percentile points", "n_participants": len(frame),
        "rank": int(rank), "n_columns": matrix.shape[1], "n_scanner_dummies": len(levels) - 1,
        "coefficient_names": names, "coefficients": coefficients.tolist() if estimable else None,
        "scanner_levels": [int(code) for code in levels],
        "reference_scanner": reference, "age_center": center,
        "condition_number": float(singular_values[0] / singular_values[-1]) if estimable else None,
        "reason": None if estimable else "rank_deficient",
        "formula": "fixed cohort percentile ~ intercept + stage05 + centered age + sex_male1 + scanner fixed effects",
    }


def scanner_block_draws(scanner_codes, n_draws=2000, rng_seed=20261004):
    """In-memory row indices only; each selected row contains all five seeds."""
    if not isinstance(n_draws, (int, np.integer)) or n_draws < 1:
        raise ValueError("Bootstrap replicate count must be a positive integer")
    codes = np.asarray(scanner_codes)
    groups = [np.flatnonzero(codes == code) for code in np.unique(codes)]
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_draws):
        yield np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in groups])


def interval_summary(samples, n_requested, reasons=None):
    valid = np.asarray([value for value in samples if value is not None], dtype=float)
    return {
        "ci95": np.quantile(valid, [0.025, 0.975], method="linear").tolist() if len(valid) else None,
        "n_requested": n_requested, "n_estimable": len(valid),
        "n_nonestimable": n_requested - len(valid), "nonestimable_reasons": reasons or {},
    }


def false_negative_summary(frame):
    eligible = tested_mask(frame.scanner_code.to_numpy())
    result = {}
    for stage in (1, 2):
        member = frame[STAGE].to_numpy() == stage
        errors = frame.loc[eligible & member, ERROR_COLUMNS].to_numpy()
        counts = errors.sum(axis=0).astype(int)
        rates = errors.mean(axis=0) if len(errors) else None
        result[STAGE_LABELS[stage]] = {
            "n_tested": len(errors), "n_untested": int((member & ~eligible).sum()),
            "counts_by_seed": counts.tolist(), "rates_by_seed": rates.tolist() if rates is not None else None,
            "mean_rate": float(rates.mean()) if rates is not None else None,
            "seed_population_sd": float(rates.std(ddof=0)) if rates is not None else None,
            "unit": "rate proportion", "seed_order": list(SEEDS), "seed_sd_ddof": 0,
        }
    return result


def bootstrap_impaired(impaired, n_draws=2000, rng_seed=20261004):
    """Shared participant draws for adjusted contrast, FN means and difference.

    Eligibility must be applied to the full cohort before this function; it
    cannot be re-derived from the smaller impaired-only fold counts.
    """
    if not impaired[STAGE].isin([1, 2]).all():
        raise ValueError("Impaired bootstrap population contains controls")
    matrix, _, _, _ = design_matrix(impaired)
    y = impaired.misfit_percentile.to_numpy()
    stage = impaired[STAGE].to_numpy()
    errors = impaired[ERROR_COLUMNS].to_numpy()
    # This five-seed within-participant mean gives each participant weight one.
    mean_errors = errors.mean(axis=1)
    samples = {key: [] for key in ("adjusted_contrast", "fn_rate_05", "fn_rate_1plus", "fn_rate_difference")}
    missing = {1: 0, 2: 0}
    zero = {1: 0, 2: 0}
    rank_deficient = 0
    difference_missing = 0
    for index in scanner_block_draws(impaired.scanner_code.to_numpy(), n_draws, rng_seed):
        coefficient, _, rank, _ = np.linalg.lstsq(matrix[index], y[index], rcond=None)
        good_rank = rank == matrix.shape[1]
        samples["adjusted_contrast"].append(float(coefficient[1]) if good_rank else None)
        rank_deficient += int(not good_rank)
        rates = {}
        for group, key in ((1, "fn_rate_05"), (2, "fn_rate_1plus")):
            chosen = mean_errors[index][stage[index] == group]
            rates[group] = float(chosen.mean()) if len(chosen) else None
            missing[group] += int(not len(chosen))
            zero[group] += int(bool(len(chosen)) and not chosen.any())
            samples[key].append(rates[group])
        good_groups = all(rates[group] is not None for group in (1, 2))
        samples["fn_rate_difference"].append(rates[1] - rates[2] if good_groups else None)
        difference_missing += int(not good_groups)
    result = {
        "adjusted_contrast": interval_summary(samples["adjusted_contrast"], n_draws, {"rank_deficient": rank_deficient}),
        "fn_rate_difference": interval_summary(samples["fn_rate_difference"], n_draws, {"missing_stage": difference_missing}),
    }
    for group, key in ((1, "fn_rate_05"), (2, "fn_rate_1plus")):
        result[key] = interval_summary(samples[key], n_draws, {"missing_stage": missing[group]})
        result[key]["n_zero_event_draws"] = zero[group]
    return result


def bootstrap_stage_medians(frame, n_draws=2000, rng_seed=20261004):
    # A separately initialized RNG with the same seed, for a distinct population.
    stage = frame[STAGE].to_numpy()
    percentile = frame.misfit_percentile.to_numpy()
    samples = {group: [] for group in range(3)}
    for index in scanner_block_draws(frame.scanner_code.to_numpy(), n_draws, rng_seed):
        for group in range(3):
            values = percentile[index][stage[index] == group]
            samples[group].append(float(np.median(values)) if len(values) else None)
    return {
        str(group): interval_summary(values, n_draws, {"missing_stage": sum(value is None for value in values)})
        for group, values in samples.items()
    }


def support_summary(frame):
    cells = []
    for code, scanner in frame.groupby("scanner_code", sort=True):
        counts = [int((scanner[STAGE] == stage).sum()) for stage in range(3)]
        cells.append({"scanner_code": int(code), "scanner": str(scanner.scanner.iloc[0]),
                      "stage_counts": counts, "n_participants": len(scanner),
                      "both_impaired_stages_present": counts[1] > 0 and counts[2] > 0,
                      "error_eligible": len(scanner) >= 5 and len(frame) - len(scanner) >= 20})
    age = {}
    for stage in (1, 2):
        values = frame.loc[frame[STAGE] == stage, "age"].to_numpy()
        age[STAGE_LABELS[stage]] = {
            "n_participants": len(values), "mean": float(values.mean()),
            "sd_sample": float(values.std(ddof=1)) if len(values) > 1 else None,
            "minimum": float(values.min()), "maximum": float(values.max()),
            "quartiles": np.quantile(values, [.25, .5, .75], method="linear").tolist(),
        }
    lower = max(item["minimum"] for item in age.values())
    upper = min(item["maximum"] for item in age.values())
    for stage in (1, 2):
        values = frame.loc[frame[STAGE] == stage, "age"].to_numpy()
        age[STAGE_LABELS[stage]]["fraction_in_shared_age_range"] = float(((values >= lower) & (values <= upper)).mean())
    return {"stage_order": ["0", "0.5", "≥1"], "scanner_by_stage": cells,
            "impaired_age_by_stage": age,
            "impaired_shared_age_range": [lower, upper] if lower <= upper else None,
            "interpretation": "Observed covariate support only; adjustment does not establish causality or an independent severity marker"}


def error_metric_summary(frame):
    tested = tested_mask(frame.scanner_code.to_numpy())
    result = {key: [] for key in ("auroc", "pooled_rho", "base_acc")}
    for loss_column, error_column in zip(LOSS_COLUMNS, ERROR_COLUMNS):
        score = frame.loc[tested, loss_column].to_numpy()
        error = frame.loc[tested, error_column].to_numpy()
        n_error, n_correct = int(error.sum()), int((error == 0).sum())
        if not n_error or not n_correct:
            raise ValueError("Canonical error metrics require both error classes")
        auc = (rankdata(score)[error == 1].sum() - n_error * (n_error + 1) / 2) / (n_error * n_correct)
        result["auroc"].append(float(auc))
        result["pooled_rho"].append(float(spearmanr(score, error).statistic))
        result["base_acc"].append(float(1 - error.mean()))
    return {key: {"by_seed": values, "mean": float(np.mean(values)),
                  "population_sd": float(np.std(values, ddof=0))} for key, values in result.items()}


def analyze(frame, n_draws=2000, rng_seed=20261004):
    validation = validate_records(frame)
    eligible = tested_mask(frame.scanner_code.to_numpy())
    impaired = frame.loc[frame.label_impaired1 == 1].copy()
    if not eligible[frame.label_impaired1.to_numpy() == 1].all():
        raise ValueError("Approved common impaired bootstrap requires all impaired participants to be tested")
    primary = fit_adjusted(impaired)
    if primary["estimate"] is None:
        raise ValueError("Primary observed design is rank deficient")
    impaired_bootstrap = bootstrap_impaired(impaired, n_draws, rng_seed)
    median_bootstrap = bootstrap_stage_medians(frame, n_draws, rng_seed)
    primary["participant_ci95"] = impaired_bootstrap["adjusted_contrast"]["ci95"]
    primary["bootstrap"] = impaired_bootstrap["adjusted_contrast"]
    seed_contrasts = []
    for column in LOSS_COLUMNS:
        outcome = fixed_percentile(frame[column])[frame.label_impaired1.to_numpy() == 1]
        seed_contrasts.append(fit_adjusted(impaired, outcome)["estimate"])
    primary["seed_specific_contrasts"] = seed_contrasts
    primary["seed_population_sd"] = float(np.std(seed_contrasts, ddof=0))
    primary["seed_sd_definition"] = "Population SD (ddof=0) of five adjusted contrasts; each seed ranked against its own original full cohort. Distinct from participant CI and from the primary rank of five-seed mean loss."
    stages = []
    for group in range(3):
        selected = frame.loc[frame[STAGE] == group]
        stages.append({
            "stage_code": group, "stage_label": STAGE_LABELS[group], "n_participants": len(selected),
            "mean_loss": float(selected.misfit_mean.mean()), "median_loss": float(selected.misfit_mean.median()),
            "mean_percentile": float(selected.misfit_percentile.mean()),
            "median_percentile": float(selected.misfit_percentile.median()),
            "median_percentile_ci95": median_bootstrap[str(group)]["ci95"],
            "bootstrap": median_bootstrap[str(group)],
        })
    rates = false_negative_summary(frame)
    for label, key in (("0.5", "fn_rate_05"), ("≥1", "fn_rate_1plus")):
        rates[label]["participant_ci95"] = impaired_bootstrap[key]["ci95"]
        rates[label]["bootstrap"] = impaired_bootstrap[key]
    seed_differences = np.array(rates["0.5"]["rates_by_seed"]) - np.array(rates["≥1"]["rates_by_seed"])
    clinical_correlations = {
        "stage_rho": float(spearmanr(frame.misfit_mean, frame[STAGE]).statistic),
        "age_rho": float(spearmanr(frame.misfit_mean, frame.age).statistic),
        "sex_rho": float(spearmanr(frame.misfit_mean, frame.sex_male1).statistic),
        "stage_rho_by_seed": [float(spearmanr(frame[column], frame[STAGE]).statistic) for column in LOSS_COLUMNS],
        "age_rho_by_seed": [float(spearmanr(frame[column], frame.age).statistic) for column in LOSS_COLUMNS],
    }
    return {
        "analysis": "OASIS-3 post hoc CDR-stage follow-up", "schema_version": 1,
        "validation": validation, "stage_summaries": stages,
        "primary_adjusted_contrast": primary,
        "unadjusted_impaired_contrasts": {
            "contrast": "CDR 0.5 minus CDR ≥1", "unit": "cohort percentile points",
            "median_percentile_difference": stages[1]["median_percentile"] - stages[2]["median_percentile"],
            "mean_percentile_difference": stages[1]["mean_percentile"] - stages[2]["mean_percentile"],
        },
        "false_negative_rates": rates,
        "false_negative_difference": {
            "contrast": "CDR 0.5 minus CDR ≥1", "unit": "rate proportion",
            "estimate": float(seed_differences.mean()), "by_seed": seed_differences.tolist(),
            "seed_population_sd": float(seed_differences.std(ddof=0)), "seed_sd_ddof": 0,
            "participant_ci95": impaired_bootstrap["fn_rate_difference"]["ci95"],
            "bootstrap": impaired_bootstrap["fn_rate_difference"],
        },
        "support": support_summary(frame), "clinical_correlations": clinical_correlations,
        "canonical_error_metrics": error_metric_summary(frame),
        "bootstrap_protocol": {
            "replicates": n_draws, "rng_seed": rng_seed, "rng": "numpy.random.default_rng / PCG64",
            "unit": "participant block with all five seed measurements kept together",
            "stratification": "Within each observed scanner, sample its original participant count with replacement; scanner codes ascending, input CSV order within scanner",
            "percentile_reference": "100 * scipy.stats.rankdata(five-seed mean loss, method='average') / original full-cohort N; fixed before resampling, never re-ranked",
            "interval": "Two-sided percentile 95% CI using numpy.quantile([0.025,0.975], method='linear') on estimable draws",
            "streams": {
                "impaired": {"population_n": len(impaired), "seed": rng_seed,
                             "purpose": "Same draws for OLS refits, both FN means and FN difference"},
                "full_cohort": {"population_n": len(frame), "seed": rng_seed,
                                "purpose": "Three descriptive stage medians; separate RNG object independently initialized with the same seed, distinct population, not an independent random seed"},
            },
            "nonestimable": "All attempted draws counted; rank-deficient OLS and missing-stage rate/median draws have explicit counts. Zero-event FN draws are valid zero rates and are counted separately.",
            "conditioning": "Intervals condition on observed scanner strata, fixed trained models/five seeds and fixed original percentile reference; they do not quantify new-scanner or model-retraining uncertainty.",
            "age_center": "Original impaired-population mean fixed for all OLS refits; centering does not change the stage contrast",
        },
        "ordinary_loss_comparison": {
            "status": "separate_analysis", "reason": "This stage-profile analysis does not compare scores; the CNN CDR/scanner analyses provide the comparator results",
            "paired_auroc_claim": False, "new_training_performed": False,
        },
        "interpretation": [
            "Post hoc observational, task-conditional comparison among impaired participants; CDR defines the diagnostic target.",
            "Age, sex and scanner adjustment describes association, not causality or independent disease-severity validation.",
            "Five seeds are repeated fits for the same participants, never five times as many independent participants.",
            "False-negative error is evaluated only for eligible held-scanner predictions; untested sentinel errors are excluded.",
            "Initialization population SD is separate from the participant bootstrap confidence interval.",
        ],
    }


def method_provenance():
    """Verify the eligibility rule from released source without importing it."""
    source = METHOD_SOURCE.read_bytes()
    tree = ast.parse(source)
    collect = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "collect")
    condition = ast.dump(ast.parse("te.sum() < 5 or tr.sum() < 20", mode="eval").body)
    if not any(isinstance(node, ast.If) and ast.dump(node.test) == condition
               and any(isinstance(child, ast.Continue) for child in node.body) for node in ast.walk(collect)):
        raise ValueError("Released collect eligibility guard could not be verified")
    return {"source": str(METHOD_SOURCE.relative_to(ROOT)), "source_sha256": hashlib.sha256(source).hexdigest(),
            "verified_fold_guard": "if te.sum() < 5 or tr.sum() < 20: continue",
            "verified_by": "Python AST of released collect; source is read-only and never executed"}


def reconcile_published(result, evidence_dir):
    """Checks reference aggregates after estimates have been computed from data."""
    clinical_path = Path(evidence_dir) / "clinical_insight_panel.json"
    validation_path = Path(evidence_dir) / "oasis3_validation_unique.json"
    clinical = json.loads(clinical_path.read_text())["oasis3_clinical"]
    canonical = json.loads(validation_path.read_text())
    stages = result["stage_summaries"]
    if [row["n_participants"] for row in stages] != clinical["stage_n"]:
        raise ValueError("Published stage count reconciliation failed")
    if [round(row["median_percentile"], 1) for row in stages] != clinical["stage_median_misfit_percentile"]:
        raise ValueError("Published stage median reconciliation failed")
    correlations = result["clinical_correlations"]
    for key, source in (("stage_rho", "misfit_vs_cdr_stage"), ("age_rho", "misfit_vs_age"), ("sex_rho", "misfit_vs_sex_male1")):
        if round(correlations[key], 3) != clinical[source]["rho"]:
            raise ValueError("Published clinical correlation reconciliation failed")
    for key, source in (("stage_rho_by_seed", "per_seed_cdr_rho"), ("age_rho_by_seed", "per_seed_age_rho")):
        if [round(value, 3) for value in correlations[key]] != clinical[source]:
            raise ValueError("Published per-seed clinical correlation reconciliation failed")
    differences = {}
    for key in ("auroc", "pooled_rho", "base_acc"):
        difference = result["canonical_error_metrics"][key]["mean"] - canonical[f"{key}_mean"]
        if not np.isclose(difference, 0, atol=1e-12, rtol=0):
            raise ValueError("Canonical error metric reconciliation failed")
        differences[f"{key}_mean_delta"] = difference
    for key in ("auroc", "pooled_rho"):
        difference = result["canonical_error_metrics"][key]["population_sd"] - canonical[f"{key}_std"]
        if not np.isclose(difference, 0, atol=1e-12, rtol=0):
            raise ValueError("Canonical error SD reconciliation failed")
        differences[f"{key}_sd_delta"] = difference
    return {
        "all_passed": True, "stage_counts_match": True, "stage_medians_match_at_published_one_decimal": True,
        "clinical_correlations_match_at_published_three_decimals": True,
        "canonical_error_metric_absolute_tolerance": 1e-12, "canonical_error_metric_deltas": differences,
        "sources": [{"path": path.name, "sha256": sha256(path)} for path in (clinical_path, validation_path)],
        "calculation_policy": "Published values used only for reconciliation, never as inputs to estimates or intervals",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="External restricted CSV, read-only; never copied")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Aggregate JSON destination")
    parser.add_argument("--reference-dir", type=Path, help="Optional directory with clinical_insight_panel.json and oasis3_validation_unique.json for reconciliation")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args(argv)
    try:
        validate_distinct_output(args.input, args.output)
        before = sha256(args.input)
        frame = load_records(args.input)
        validate_records(frame, expected_n=1360, expected_counts=(976, 277, 107))
        result = analyze(frame, args.replicates, args.seed)
        result["published_reconciliation"] = (reconcile_published(result, args.reference_dir)
                                               if args.reference_dir else {"status": "not_requested"})
        result["provenance"] = {
            "restricted_source_sha256": before,
            "restricted_source_stored_in_project": False,
            "source_retention": "External source opened read-only; no participant-level output or resampling indices written",
            "script": "misfit/manuscript_analysis/analyze_cdr_stage.py", "script_sha256": sha256(Path(__file__)),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scipy": scipy.__version__},
            "ols_solver": "numpy.linalg.lstsq with rcond=None; full-column-rank required",
            "released_method": method_provenance(),
        }
        if sha256(args.input) != before:
            raise ValueError("Source changed while analysis was running")
        result["provenance"]["source_hash_unchanged_after_analysis"] = True
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        # Generated output has an explicit aggregate-only schema, never a frame dump.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print("CDR aggregate analysis completed; see JSON for source checks and optional reference reconciliation.")
        return 0
    except Exception:
        # Input parser errors can contain row data, so never render their details.
        print("CDR analysis failed: input, protocol, provenance, or aggregate reconciliation check did not pass. No row diagnostics are emitted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
