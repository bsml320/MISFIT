"""Statistical contracts for the aggregate-only CNN CDR/scanner analysis."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "misfit/manuscript_analysis/analyze_cnn_cdr_scanner.py"
if SCRIPT.exists():
    SPEC = importlib.util.spec_from_file_location("cnn_cdr_scanner", SCRIPT)
    ANALYSIS = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(ANALYSIS)
else:
    ANALYSIS = None


def synthetic_data(n=8):
    """Three seed records share a unique participant and paired score columns."""
    error = np.tile(np.arange(n) % 2, (3, 1)).astype(bool)
    scores = np.tile(np.linspace(0, 1, n), (3, 1))
    return {
        "scanners": np.array(["SyntheticScanner"]),
        "seeds": np.array([42, 43, 44]),
        "site": np.zeros(n, dtype=int),
        "cdr": np.full(n, 0.5),
        "tested": np.ones((3, n), dtype=bool),
        "error": error,
        "misfit": scores,
        "ordinary": scores.copy(),
    }


class CnnCdrScannerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ANALYSIS, "The approved analysis module must exist")

    def test_auc_ties_direction_and_degenerate_outcomes(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(ANALYSIS.weighted_auc(y, [0, 1, 1, 2]), 0.875)
        self.assertEqual(ANALYSIS.weighted_auc(y, [0, 1, 2, 3]), 1.0)
        self.assertEqual(ANALYSIS.weighted_auc(y, [3, 2, 1, 0]), 0.0)
        self.assertEqual(ANALYSIS.weighted_auc(y, [1, 1, 1, 1]), 0.5)
        self.assertTrue(np.isnan(ANALYSIS.weighted_auc([1, 1], [0, 1])))
        self.assertTrue(np.isnan(ANALYSIS.weighted_auc(y, [0, 1, 2, 3], [1, 2, 0, 0])))

    def test_weighted_auc_matches_explicit_resampling_with_ties(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        score = np.array([0, 0, 1, 1, 2, 2])
        weights = np.array([[2, 1, 0, 3, 2, 1], [0, 4, 2, 0, 1, 2]])
        actual = ANALYSIS.weighted_auc(y, score, weights)
        expected = []
        for row in weights:
            expanded = np.repeat(np.arange(y.size), row)
            ey, es = y[expanded], score[expanded]
            differences = es[ey == 1, None] - es[None, ey == 0]
            expected.append(np.mean((differences > 0) + 0.5 * (differences == 0)))
        np.testing.assert_allclose(actual, expected)

    def test_unique_alignment_reorders_and_validates_labels_scanners(self):
        cnn = {"sessions": np.array(["A_session", "B_session"]),
               "label": np.array([0, 1]), "site": np.array([0, 1]),
               "scanners": np.array(["One", "Two"]),
               "tested": np.ones((3, 2), dtype=bool)}
        clinical = {"sessions": np.array(["B_session", "A_session"]),
                    "label": np.array([1, 0]), "scanner": np.array(["Two", "One"]),
                    "cdr_raw": np.array([0.5, 0.0]),
                    "tested_per_seed": np.ones((5, 2), dtype=bool)}
        np.testing.assert_array_equal(ANALYSIS.align_clinical(cnn, clinical), [0.0, 0.5])
        for field in ("sessions", "label", "scanner"):
            broken = {key: val.copy() for key, val in clinical.items()}
            broken[field][0] = broken[field][1]
            with self.assertRaises(ValueError):
                ANALYSIS.align_clinical(cnn, broken)
        clinical["tested_per_seed"][0, 0] = False
        with self.assertRaises(ValueError):
            ANALYSIS.align_clinical(cnn, clinical)

    def test_fold_scores_are_finite_mean_and_first_in_saved_order(self):
        folds = np.array([[[np.nan, 1, 3], [2, np.nan, np.nan], [4, 5, 9]]])
        mean, ordinary = ANALYSIS.source_fold_scores(folds)
        np.testing.assert_allclose(mean, [[3, 3, 6]])
        np.testing.assert_allclose(ordinary, [[2, 1, 3]])
        with self.assertRaises(ValueError):
            ANALYSIS.source_fold_scores(np.full((1, 2, 3), np.nan))

    def test_identical_paired_scores_have_exact_zero_delta_interval(self):
        results = ANALYSIS.analyze(synthetic_data(), attempts=200, rng_seed=81)
        stage = results["stages"][1]
        self.assertEqual(stage["name"], "CDR 0.5")
        self.assertEqual(stage["summary"]["delta_mean"], 0.0)
        self.assertEqual(stage["summary"]["bootstrap"]["delta_ci95"], [0.0, 0.0])
        self.assertEqual(stage["cells"][0]["summary"]["bootstrap"]["delta_ci95"], [0.0, 0.0])

    def test_original_eligibility_and_weights_remain_fixed(self):
        data = synthetic_data()
        # One original seed has no errors and is never included in the estimand.
        data["error"][2] = False
        # Add a tiny scanner: it must be visible but excluded for all seeds.
        for field in ("error", "tested", "misfit", "ordinary"):
            data[field] = np.concatenate((data[field], data[field][:, :4]), axis=1)
        data["scanners"] = np.array(["SyntheticScanner", "SmallScanner"])
        data["site"] = np.r_[np.zeros(8, dtype=int), np.ones(4, dtype=int)]
        data["cdr"] = np.full(12, 0.5)
        result = ANALYSIS.analyze(data, attempts=100, rng_seed=10)["stages"][1]
        self.assertEqual(result["summary"]["eligible_scanner_seed_cells"], 2)
        self.assertEqual(result["cells"][0]["seeds"][2]["reason"], "single outcome")
        self.assertEqual(result["cells"][1]["summary"]["eligible_scanner_seed_cells"], 0)
        self.assertIsNone(result["cells"][1]["summary"]["misfit_mean"])
        self.assertEqual(result["summary"]["misfit_mean"], ANALYSIS.weighted_auc(data["error"][0, :8], data["misfit"][0, :8]))

    def test_invalid_whole_stage_draws_are_discarded_without_reweighting(self):
        data = synthetic_data()
        data["scanners"] = np.array(["A", "B"])
        data["site"] = np.repeat([0, 1], 8)
        data["cdr"] = np.full(16, 0.5)
        for field in ("error", "tested", "misfit", "ordinary"):
            data[field] = np.tile(data[field], (1, 2))
        data["error"][:] = False
        data["error"][0, [0, 8]] = True
        data["error"][1, [1, 9]] = True
        data["error"][2, [2, 10]] = True
        result = ANALYSIS.analyze(data, attempts=300, rng_seed=122)
        repeat = ANALYSIS.analyze(data, attempts=300, rng_seed=122)
        self.assertEqual(result, repeat)
        stage = result["stages"][1]
        boot = stage["summary"]["bootstrap"]
        self.assertEqual(boot["attempted"], 300)
        # Each scanner needs three distinct rare cases; the stage needs all six.
        self.assertGreater(boot["invalid"], 0)
        self.assertGreater(boot["valid"], 0)
        self.assertEqual(boot["valid"] + boot["invalid"], 300)
        for cell in stage["cells"]:
            self.assertGreater(cell["summary"]["bootstrap"]["valid"], boot["valid"])

    def test_untested_participants_are_counted_but_never_scored(self):
        data = synthetic_data(n=10)
        data["tested"][:, -2:] = False
        data["misfit"][:, -2:] = 1e90
        data["ordinary"][:, -2:] = -1e90
        result = ANALYSIS.analyze(data, attempts=100, rng_seed=21)["stages"][1]
        cell = result["cells"][0]
        self.assertEqual(cell["n_participants"], 10)
        self.assertEqual([s["n_tested"] for s in cell["seeds"]], [8, 8, 8])
        self.assertEqual(cell["summary"]["delta_mean"], 0.0)

    def test_outputs_are_aggregate_only_and_json_has_no_nonfinite_literals(self):
        data = synthetic_data()
        data["sessions"] = np.array([f"PRIVATE_SUBJECT_{i}" for i in range(8)])
        results = ANALYSIS.analyze(data, attempts=100, rng_seed=19)
        serialized = json.dumps(results, allow_nan=False)
        self.assertNotIn("PRIVATE_SUBJECT", serialized)
        for forbidden in ("sessions", "cdr_raw", '"pred"', '"scores"'):
            self.assertNotIn(forbidden, serialized)
        report = ANALYSIS.render_report(results)
        self.assertIn("5000", ANALYSIS.BOOTSTRAP_DEFINITION)
        self.assertIn("fixed", report.lower())
        self.assertNotIn("PRIVATE_SUBJECT", report)

    def test_hash_mismatch_is_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            ANALYSIS.verify_hash(SCRIPT, "0" * 64)


if __name__ == "__main__":
    unittest.main()
