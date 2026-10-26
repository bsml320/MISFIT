"""Statistical contracts for different-score/target-seed CNN analysis."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "misfit/manuscript_analysis/analyze_cnn_crossseed.py"
sys.path.insert(0, str(SCRIPT.parent))
if SCRIPT.exists():
    SPEC = importlib.util.spec_from_file_location("cnn_crossseed", SCRIPT)
    ANALYSIS = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(ANALYSIS)
else:
    ANALYSIS = None

METHODS = ("misfit_ce", "firstfinite_ce", "firstfinite_aum", "firstfinite_el2n",
           "crossrefit_aum", "crossrefit_el2n")


def synthetic_data(n=8, identical=False):
    error = np.array([np.arange(n) % 2, np.arange(n) % 3 == 0,
                      np.arange(n) % 4 < 2], dtype=bool)
    scores = np.array([np.arange(n), np.arange(n)[::-1],
                       np.roll(np.arange(n), 3)], dtype=float)
    return {"scanners": np.array(["SyntheticScanner"]),
            "seeds": np.array([42, 43, 44]), "site": np.zeros(n, dtype=int),
            "cdr": np.full(n, 0.5), "tested": np.ones((3, n), dtype=bool),
            "error": error,
            "scores": {method: scores.copy() if identical else np.roll(scores, i, axis=1)
                       for i, method in enumerate(METHODS)}}


def hand_auc(y, score, weights=None):
    """Independent pairwise U statistic, with multiplicities expanded."""
    if weights is not None:
        index = np.repeat(np.arange(len(y)), weights)
        y, score = np.asarray(y)[index], np.asarray(score)[index]
    y, score = np.asarray(y, dtype=bool), np.asarray(score)
    if not np.any(y) or np.all(y):
        return np.nan
    difference = score[y, None] - score[None, ~y]
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


class CnnCrossseedTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ANALYSIS, "The approved different-seed analysis must exist")

    def test_six_ordered_pairs_use_score_seed_and_target_seed_separately(self):
        data = synthetic_data()
        result = ANALYSIS.analyze(data, attempts=40, rng_seed=17)
        cell = result["stages"][1]["cells"][0]
        pairs = cell["seed_pairs"]
        self.assertEqual([(p["score_seed"], p["error_seed"]) for p in pairs],
                         [(42, 43), (42, 44), (43, 42), (43, 44), (44, 42), (44, 43)])
        expected = {method: [] for method in METHODS}
        for pair in pairs:
            s, t = pair["score_seed"] - 42, pair["error_seed"] - 42
            for method in METHODS:
                auc = hand_auc(data["error"][t], data["scores"][method][s])
                self.assertEqual(pair["auroc"][method], auc)
                expected[method].append(auc)
            for method in METHODS[1:]:
                self.assertEqual(pair["delta"][method],
                                 pair["auroc"][METHODS[0]] - pair["auroc"][method])
        summary = result["stages"][1]["summary"]
        self.assertEqual(summary["eligible_scanner_pair_cells"], 6)
        for method in METHODS:
            self.assertEqual(summary["methods"][method]["mean"], np.mean(expected[method]))
            self.assertEqual(summary["methods"][method]["population_sd"], np.std(expected[method]))

    def test_reused_weighted_auc_ties_and_multiplicities_match_hand_statistic(self):
        from misfit.manuscript_analysis.analyze_cnn_cdr_scanner import weighted_auc
        self.assertIs(ANALYSIS.weighted_auc, weighted_auc)
        y, score = np.array([0, 1, 0, 1, 0, 1]), np.array([0, 0, 1, 1, 2, 2])
        weights = np.array([[2, 1, 0, 3, 2, 1], [0, 4, 2, 0, 1, 2]])
        np.testing.assert_allclose(ANALYSIS.weighted_auc(y, score, weights),
                                   [hand_auc(y, score, row) for row in weights])
        self.assertTrue(np.isnan(ANALYSIS.weighted_auc(np.ones(6), score)))

    def test_fold_reduction_shares_first_finite_fold_and_preserves_aum_direction(self):
        ce = np.array([[[np.nan, 1, 3], [2, np.nan, np.nan], [4, 5, 9]]])
        aum, el2n = ce - 5, ce / 10
        scores = ANALYSIS.build_scores(ce, aum, el2n)
        np.testing.assert_allclose(scores["misfit_ce"], [[3, 3, 6]])
        np.testing.assert_allclose(scores["firstfinite_ce"], [[2, 1, 3]])
        np.testing.assert_allclose(scores["firstfinite_aum"], [[-3, -4, -2]])
        np.testing.assert_allclose(scores["crossrefit_aum"], [[-2, -2, 1]])
        np.testing.assert_allclose(scores["firstfinite_el2n"], [[0.2, 0.1, 0.3]])
        np.testing.assert_allclose(scores["crossrefit_el2n"], [[0.3, 0.3, 0.6]])

    def test_fold_masks_must_match_and_no_infinite_or_empty_source_scores(self):
        ce = np.array([[[np.nan, 1], [2, 3]]], dtype=float)
        for bad in (np.array([[[0, 1], [2, 3]]], dtype=float),
                    np.array([[[np.nan, np.inf], [2, 3]]], dtype=float),
                    np.ones((1, 3, 2))):
            with self.subTest(bad_shape=bad.shape), self.assertRaises(ValueError):
                ANALYSIS.build_scores(ce, bad, ce)
        with self.assertRaises(ValueError):
            ANALYSIS.build_scores(np.full((3, 2, 4), np.nan),
                                  np.full((3, 2, 4), np.nan), np.full((3, 2, 4), np.nan))

    def test_identical_methods_have_zero_paired_delta_and_interval(self):
        result = ANALYSIS.analyze(synthetic_data(identical=True), attempts=100, rng_seed=81)
        stage = result["stages"][1]
        for summary in (stage["summary"], stage["cells"][0]["summary"]):
            for method in METHODS[1:]:
                self.assertEqual(summary["deltas"][method],
                                 {"mean": 0.0, "population_sd": 0.0, "ci95": [0.0, 0.0]})

    def test_original_target_eligibility_and_untested_exclusion(self):
        data = synthetic_data(10)
        data["tested"][:, -2:] = False
        data["error"][2] = False
        for score in data["scores"].values():
            score[:, -2:] = np.nan
        result = ANALYSIS.analyze(data, attempts=50, rng_seed=7)["stages"][1]
        self.assertEqual(result["n_participants"], 10)
        self.assertEqual(result["n_tested_participants"], 8)
        self.assertEqual(result["summary"]["eligible_scanner_pair_cells"], 4)
        for pair in result["cells"][0]["seed_pairs"]:
            self.assertEqual(pair["n_tested"], 8)
            if pair["error_seed"] == 44:
                self.assertEqual(pair["reason"], "single outcome")
                self.assertTrue(all(v is None for v in pair["auroc"].values()))
        small = ANALYSIS.analyze(synthetic_data(7), attempts=50)["stages"][1]
        self.assertEqual(small["summary"]["eligible_scanner_pair_cells"], 0)
        self.assertEqual(small["summary"]["bootstrap"]["attempted"], 0)
        self.assertTrue(all(p["reason"] == "n < 8" for p in small["cells"][0]["seed_pairs"]))

    def test_invalid_inputs_fail_before_resampling(self):
        for field in ("tested", "scores", "error", "seeds", "cdr", "site"):
            data = synthetic_data()
            if field == "tested":
                data[field][1, 0] = False
            elif field == "scores":
                data[field]["firstfinite_aum"][0, 0] = np.nan
            elif field == "error":
                data[field] = data[field].astype(float)
                data[field][0, 0] = np.nan
            elif field == "seeds":
                data[field] = np.array([42, 42, 44])
            elif field == "cdr":
                data[field][0] = np.nan
            else:
                data[field][0] = 10
            with self.subTest(field=field), self.assertRaises(ValueError):
                ANALYSIS.analyze(data, attempts=20)
        with self.assertRaises(ValueError):
            ANALYSIS.analyze(synthetic_data(), attempts=0)

    def test_joint_bootstrap_matches_independent_expansion_and_complete_stage_rejection(self):
        data = synthetic_data()
        data["scanners"] = np.array(["A", "B"])
        data["site"] = np.repeat([0, 1], 8)
        data["cdr"] = np.full(16, 0.5)
        for field in ("tested", "error"):
            data[field] = np.tile(data[field], (1, 2))
        data["scores"] = {k: np.tile(v, (1, 2)) for k, v in data["scores"].items()}
        data["error"][:] = False
        for seed in range(3):
            data["error"][seed, [seed, 8 + seed]] = True
        attempts, seed = 180, 122
        result = ANALYSIS.analyze(data, attempts=attempts, rng_seed=seed)
        self.assertEqual(result, ANALYSIS.analyze(data, attempts=attempts, rng_seed=seed))
        stage = result["stages"][1]
        rng = np.random.Generator(np.random.PCG64(seed))
        scanner_draws = []
        for scanner in range(2):
            weights = rng.multinomial(8, np.full(8, 1 / 8), size=attempts)
            index = np.arange(scanner * 8, (scanner + 1) * 8)
            draws = np.array([[[hand_auc(data["error"][t, index],
                                        data["scores"][m][s, index], w) for m in METHODS]
                               for s in range(3) for t in range(3) if s != t] for w in weights])
            scanner_draws.append(draws)
            valid = np.all(np.isfinite(draws), axis=(1, 2))
            self.assertEqual(stage["cells"][scanner]["summary"]["bootstrap"]["valid"], int(valid.sum()))
        combined = np.concatenate(scanner_draws, axis=1)
        valid = np.all(np.isfinite(combined), axis=(1, 2))
        self.assertGreater(valid.sum(), 0)
        self.assertLess(valid.sum(), attempts)
        self.assertEqual(stage["summary"]["bootstrap"]["valid"], int(valid.sum()))
        means = combined[valid].mean(axis=1)
        for m, method in enumerate(METHODS):
            np.testing.assert_allclose(stage["summary"]["methods"][method]["ci95"],
                                       np.percentile(means[:, m], [2.5, 97.5], method="linear"))
            if m:
                np.testing.assert_allclose(stage["summary"]["deltas"][method]["ci95"],
                                           np.percentile(means[:, 0] - means[:, m], [2.5, 97.5], method="linear"))
        for omitted in stage["leave_one_scanner_out"]:
            retained = stage["cells"][1 if omitted["omitted_scanner"] == "A" else 0]
            self.assertEqual(omitted["eligible_scanner_pair_cells"], 6)
            for method in METHODS:
                self.assertEqual(omitted["methods"][method]["mean"],
                                 retained["summary"]["methods"][method]["mean"])
            self.assertNotIn("bootstrap", omitted)

    def test_report_and_json_are_aggregate_only_and_state_inferential_boundary(self):
        data = synthetic_data()
        data["sessions"] = np.array([f"PRIVATE_SUBJECT_{i}" for i in range(8)])
        result = ANALYSIS.analyze(data, attempts=50, rng_seed=19)
        serialized = json.dumps(result, allow_nan=False)
        report = ANALYSIS.render_report(result)
        for text in (serialized, report):
            self.assertNotIn("PRIVATE_SUBJECT", text)
            self.assertNotIn("cdr_raw", text)
        for phrase in ("epoch-20", "saved AUM", "producer", "unadjusted", "pointwise",
                       "descriptive", "different", "fixed", "PCG64"):
            self.assertIn(phrase, report)
        self.assertIn("42 -> 43", report)
        self.assertIn("CDR 0.5 (primary", report)

    def test_shared_verified_loader_and_hash_rejection_are_retained(self):
        from misfit.manuscript_analysis.analyze_cnn_cdr_scanner import load_verified_inputs, verify_hash
        self.assertIs(ANALYSIS.load_same_seed_inputs, load_verified_inputs)
        self.assertIs(ANALYSIS.verify_hash, verify_hash)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            ANALYSIS.load_verified_inputs(SCRIPT, SCRIPT)


if __name__ == "__main__":
    unittest.main()
