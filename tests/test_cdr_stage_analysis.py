"""Synthetic checks only: restricted participant records never enter tests."""

import importlib.util
import hashlib
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "misfit/manuscript_analysis/analyze_cdr_stage.py"


def synthetic_records():
    rng = np.random.default_rng(199)
    n = 122
    stage = np.r_[np.tile([0, 1, 2, 1], 30), [0, 0]]
    losses = rng.uniform(0.01, 2, (n, 5))
    mean = losses.mean(axis=1)
    codes = np.r_[np.repeat([0, 1, 2], 40), [3, 3]]
    frame = pd.DataFrame({
        "session": [f"SYNTHETIC_SECRET_{i:04d}" for i in range(n)],
        "scanner": np.array(["Alpha", "Beta", "Gamma", "Tiny"])[codes],
        "scanner_code": codes,
        "label_impaired1": (stage > 0).astype(int),
        "cdr_raw": np.array([0, 0.5, 1])[stage],
        "cdr_stage_0_05_1plus": stage,
        "age": rng.uniform(50, 90, n),
        "sex_male1": rng.integers(0, 2, n),
        "misfit_mean": mean,
        "misfit_percentile": 100 * rankdata(mean, method="average") / n,
    })
    # The participant extraction convention is the prefix before '_MR_'.
    frame["session"] = [f"SYNTHETIC_SECRET_{i:04d}_MR_d0000" for i in range(n)]
    for j, seed in enumerate(range(42, 47)):
        frame[f"misfit_seed{seed}"] = losses[:, j]
    for j, seed in enumerate(range(42, 47)):
        frame[f"err_seed{seed}"] = rng.integers(0, 2, n)
        frame.loc[n - 2:, f"err_seed{seed}"] = 1
    return frame


class CDRStageAnalysis(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), "Aggregate CDR analysis implementation is missing")
        spec = importlib.util.spec_from_file_location("cdr_stage_analysis", SCRIPT)
        self.analysis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.analysis)
        self.frame = synthetic_records()

    def test_average_rank_ties_keep_original_one_over_n_convention(self):
        np.testing.assert_array_equal(
            self.analysis.fixed_percentile([1, 1, 3, 8]), [37.5, 37.5, 75, 100]
        )

    def test_input_roundtrip_preserves_exact_five_seed_mean(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.csv"
            self.frame.to_csv(path, index=False)
            loaded = self.analysis.load_records(path)
        checks = self.analysis.validate_records(loaded)
        self.assertEqual(checks["n_participants"], 122)
        self.assertTrue(checks["five_seed_mean_exact"])
        self.assertTrue(checks["percentile_exact"])

    def test_input_validation_rejects_each_bad_definition_without_row_echo(self):
        cases = [
            ("age", np.inf), ("sex_male1", 2), ("label_impaired1", 1),
            ("cdr_raw", 0.5), ("cdr_stage_0_05_1plus", 4),
            ("err_seed42", -1), ("misfit_mean", 10), ("misfit_percentile", 0),
            ("scanner", "Beta"),
        ]
        for column, value in cases:
            with self.subTest(column=column):
                changed = self.frame.copy()
                changed.loc[0, column] = value
                with self.assertRaises(ValueError) as raised:
                    self.analysis.validate_records(changed)
                self.assertNotIn("SYNTHETIC_SECRET", str(raised.exception))
        changed = self.frame.copy()
        changed.loc[1, "session"] = changed.loc[0, "session"]
        with self.assertRaises(ValueError):
            self.analysis.validate_records(changed)
        changed = self.frame.copy()
        changed.loc[1, "session"] = changed.loc[0, "session"].replace("d0000", "d0001")
        with self.assertRaises(ValueError):
            self.analysis.validate_records(changed)

    def test_fold_eligibility_uses_test_and_training_size_not_error(self):
        mask = self.analysis.tested_mask(self.frame["scanner_code"].to_numpy())
        self.assertEqual(int(mask.sum()), 120)
        self.assertFalse(mask[-1])
        # Eligibility is independent of a sentinel-like error among valid folds.
        self.frame.loc[0, [f"err_seed{s}" for s in range(42, 47)]] = 1
        self.assertTrue(mask[0])
        tiny_train = np.r_[np.zeros(40, dtype=int), np.ones(19, dtype=int)]
        tiny_mask = self.analysis.tested_mask(tiny_train)
        self.assertFalse(tiny_mask[:40].any())
        self.assertTrue(tiny_mask[40:].all())

    def test_untested_error_sentinel_is_not_included_in_fn(self):
        # Make the two untested records impaired to exercise the generic guard.
        self.frame.loc[120:, "cdr_stage_0_05_1plus"] = 1
        self.frame.loc[120:, "label_impaired1"] = 1
        self.frame.loc[120:, "cdr_raw"] = 0.5
        result = self.analysis.false_negative_summary(self.frame)
        self.assertEqual(result["0.5"]["n_tested"], 60)
        self.assertEqual(result["0.5"]["n_untested"], 2)
        errors = self.frame.loc[:119, [f"err_seed{s}" for s in range(42, 47)]].to_numpy()
        stage = self.frame.loc[:119, "cdr_stage_0_05_1plus"].to_numpy()
        self.assertEqual(result["0.5"]["counts_by_seed"], errors[stage == 1].sum(axis=0).tolist())

    def test_adjusted_ols_recovers_known_contrast_and_reports_rank(self):
        impaired = self.frame[self.frame.label_impaired1 == 1].copy()
        y = (13 + 7 * (impaired.cdr_stage_0_05_1plus == 1)
             + 0.3 * impaired.age - 2 * impaired.sex_male1
             + 4 * (impaired.scanner_code == 1) - 3 * (impaired.scanner_code == 2))
        model = self.analysis.fit_adjusted(impaired, y.to_numpy())
        self.assertAlmostEqual(model["estimate"], 7, places=11)
        self.assertEqual(model["rank"], 6)
        self.assertEqual(model["n_columns"], 6)
        self.assertEqual(model["n_scanner_dummies"], 2)
        impaired["sex_male1"] = 1
        deficient = self.analysis.fit_adjusted(impaired, y.to_numpy())
        self.assertIsNone(deficient["estimate"])
        self.assertEqual(deficient["reason"], "rank_deficient")

    def test_scanner_stratified_draws_preserve_blocks_and_are_reproducible(self):
        codes = self.frame.scanner_code.to_numpy()
        first = list(self.analysis.scanner_block_draws(codes, 10, 20261004))
        second = list(self.analysis.scanner_block_draws(codes, 10, 20261004))
        errors = self.frame[[f"err_seed{s}" for s in range(42, 47)]].to_numpy()
        for indices, repeated in zip(first, second):
            np.testing.assert_array_equal(indices, repeated)
            self.assertEqual(len(indices), len(self.frame))
            np.testing.assert_array_equal(np.bincount(codes[indices]), np.bincount(codes))
            self.assertEqual(errors[indices].shape, (122, 5))
            for block, index in zip(errors[indices], indices):
                np.testing.assert_array_equal(block, errors[index])

    def test_same_impaired_draws_define_fn_means_difference_and_adjusted_ci(self):
        impaired = self.frame[self.frame.label_impaired1 == 1].reset_index(drop=True)
        impaired["misfit_percentile"] = (13 + 7 * (impaired.cdr_stage_0_05_1plus == 1)
            + 0.3 * impaired.age - 2 * impaired.sex_male1 + 4 * impaired.scanner_code)
        result = self.analysis.bootstrap_impaired(impaired, 80, 20261004)
        np.testing.assert_allclose(result["adjusted_contrast"]["ci95"], [7, 7], atol=1e-10)
        means = impaired[[f"err_seed{s}" for s in range(42, 47)]].to_numpy().mean(axis=1)
        stage = impaired.cdr_stage_0_05_1plus.to_numpy()
        rate05, rate1, delta = [], [], []
        for index in self.analysis.scanner_block_draws(impaired.scanner_code.to_numpy(), 80, 20261004):
            first = means[index][stage[index] == 1].mean()
            second = means[index][stage[index] == 2].mean()
            rate05.append(first)
            rate1.append(second)
            delta.append(first - second)
        for key, values in (("fn_rate_05", rate05), ("fn_rate_1plus", rate1), ("fn_rate_difference", delta)):
            np.testing.assert_allclose(result[key]["ci95"], np.quantile(values, [.025, .975], method="linear"))
        self.assertEqual(result, self.analysis.bootstrap_impaired(impaired, 80, 20261004))

    def test_nonestimable_and_zero_event_draws_are_accounted_for(self):
        impaired = self.frame[self.frame.label_impaired1 == 1].copy()
        impaired["sex_male1"] = 1
        impaired[[f"err_seed{s}" for s in range(42, 47)]] = 0
        result = self.analysis.bootstrap_impaired(impaired, 30, 20261004)
        self.assertEqual(result["adjusted_contrast"]["n_nonestimable"], 30)
        self.assertIsNone(result["adjusted_contrast"]["ci95"])
        self.assertEqual(result["fn_rate_05"]["n_nonestimable"], 0)
        self.assertEqual(result["fn_rate_05"]["n_zero_event_draws"], 30)
        self.assertEqual(result["fn_rate_05"]["ci95"], [0.0, 0.0])

    def test_missing_stage_resamples_are_counted_in_both_model_and_fn(self):
        impaired = self.frame[self.frame.label_impaired1 == 1].copy()
        impaired["cdr_stage_0_05_1plus"] = 1
        impaired.loc[impaired.index[0], "cdr_stage_0_05_1plus"] = 2
        result = self.analysis.bootstrap_impaired(impaired, 60, 20261004)
        missing = result["fn_rate_1plus"]["n_nonestimable"]
        self.assertGreater(missing, 0)
        self.assertLess(missing, 60)
        self.assertEqual(missing, result["adjusted_contrast"]["n_nonestimable"])
        self.assertEqual(missing, result["fn_rate_difference"]["n_nonestimable"])
        for summary in result.values():
            self.assertEqual(summary["n_requested"], summary["n_estimable"] + summary["n_nonestimable"])

    def test_descriptive_bootstrap_retains_full_cohort_fixed_rank_reference(self):
        result = self.analysis.bootstrap_stage_medians(self.frame, 50, 20261004)
        samples = {str(stage): [] for stage in range(3)}
        for index in self.analysis.scanner_block_draws(self.frame.scanner_code.to_numpy(), 50, 20261004):
            sample = self.frame.iloc[index]
            for stage in range(3):
                samples[str(stage)].append(float(sample.loc[sample.cdr_stage_0_05_1plus == stage, "misfit_percentile"].median()))
        for stage in range(3):
            np.testing.assert_allclose(result[str(stage)]["ci95"], np.quantile(samples[str(stage)], [.025, .975], method="linear"))

    def test_aggregate_output_contains_no_sessions_rows_or_resampling_indices(self):
        result = self.analysis.analyze(self.frame, n_draws=30, rng_seed=20261004)
        encoded = json.dumps(result, allow_nan=False)
        self.assertNotIn("SYNTHETIC_SECRET", encoded)
        for forbidden in ("bootstrap_indices", "participant_records", "session_ids", "predictions"):
            self.assertNotIn(f'"{forbidden}":', encoded)
        self.assertEqual(result["validation"]["n_participants"], 122)
        self.assertEqual(result["primary_adjusted_contrast"]["n_participants"], 90)
        self.assertEqual(result["ordinary_loss_comparison"]["status"], "separate_analysis")
        self.assertFalse(result["ordinary_loss_comparison"]["paired_auroc_claim"])
        self.assertEqual(result["bootstrap_protocol"]["replicates"], 30)
        self.assertEqual(result["bootstrap_protocol"]["streams"]["impaired"]["population_n"], 90)
        self.assertEqual(result["bootstrap_protocol"]["streams"]["full_cohort"]["population_n"], 122)

    def test_generic_cli_failure_keeps_private_values_out_of_logs_and_source_intact(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "synthetic.csv"
            destination = Path(temporary) / "aggregate.json"
            source.write_text("private_field\nSYNTHETIC_RESTRICTED_VALUE\n")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            process = subprocess.run(
                [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(destination)],
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertNotIn("SYNTHETIC_RESTRICTED_VALUE", process.stdout + process.stderr)
            self.assertNotIn("Traceback", process.stdout + process.stderr)
            self.assertFalse(destination.exists())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)

    def test_output_alias_guard_rejects_real_hardlinks_and_symlinks(self):
        self.assertTrue(hasattr(self.analysis, "validate_distinct_output"),
                        "Input/output alias guard is missing")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "synthetic.csv"
            source.write_text("synthetic read-only source\n")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            hardlink = Path(temporary) / "hardlink.json"
            symlink = Path(temporary) / "symlink.json"
            os.link(source, hardlink)
            symlink.symlink_to(source)
            for output in (source, hardlink, symlink):
                with self.subTest(alias=output.name):
                    with self.assertRaisesRegex(ValueError, "different files"):
                        self.analysis.validate_distinct_output(source, output)
                    with patch.object(self.analysis, "sha256") as hashing, \
                            patch.object(self.analysis, "load_records") as reading, \
                            contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(self.analysis.main([
                            "--input", str(source), "--output", str(output),
                        ]), 1)
                        hashing.assert_not_called()
                        reading.assert_not_called()
                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            # Distinct existing and new destinations remain valid.
            distinct = Path(temporary) / "distinct.json"
            distinct.write_text("unrelated aggregate output\n")
            self.analysis.validate_distinct_output(source, distinct)
            self.analysis.validate_distinct_output(source, Path(temporary) / "new.json")

    def test_extended_release_schema_checks_predictions_before_selecting_columns(self):
        frame = synthetic_records()
        tested = self.analysis.tested_mask(frame.scanner_code.to_numpy())
        for seed in self.analysis.SEEDS:
            frame[f"pred_seed{seed}"] = np.where(tested, frame.label_impaired1.to_numpy().astype(int) ^ frame[f"err_seed{seed}"].to_numpy().astype(int), -1)
            frame[f"tested_seed{seed}"] = tested
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.csv"
            frame.to_csv(path, index=False)
            loaded = self.analysis.load_records(path)
            self.assertEqual(list(loaded.columns), self.analysis.COLUMNS)
            np.testing.assert_array_equal(loaded.misfit_mean, frame.misfit_mean)
            frame.loc[0, "tested_seed42"] = not bool(tested[0])
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "eligibility"):
                self.analysis.load_records(path)

    def test_released_collect_guard_is_checked_without_execution(self):
        provenance = self.analysis.method_provenance()
        self.assertEqual(provenance["verified_fold_guard"], "if te.sum() < 5 or tr.sum() < 20: continue")
        self.assertEqual(len(provenance["source_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
