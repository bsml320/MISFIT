"""Synthetic tests for the predeclared auxiliary MRI validation runner.

These tests never load the MRI input. All fixtures are deterministic toy data.
"""

import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    rv = importlib.import_module("misfit.rank_validation.real_validation.run_validation")
except ModuleNotFoundError as exc:
    if exc.name != "run_validation":
        raise
    rv = None


class ValidationTests(unittest.TestCase):
    def helper(self, name):
        self.assertIsNotNone(rv, "The controlled validation runner is not implemented")
        function = getattr(rv, name, None)
        self.assertTrue(callable(function), name + " is not implemented")
        return function

    @staticmethod
    def data():
        k = np.arange(36, dtype=float)
        x = np.column_stack((np.sin(k * 0.7), np.cos(k * 0.3), (k % 5 - 2) / 3))
        y = (np.sin(k * 0.7) + 0.4 * np.cos(k * 0.3) > 0).astype(int)
        site = np.repeat(np.arange(3), 12)
        cdr = np.where(y, np.where(k.astype(int) % 2, 0.5, 1.0), 0.0)
        return {"X": x, "y": y, "site": site, "cdr_raw": cdr}

    def test_global_normalization_retains_one_constant_penalized_intercept(self):
        normalize = self.helper("prepare_features")
        x = np.array([[3.0, 4.0], [0.0, 1.0], [1.0, -2.0]])
        z, scale = normalize(x)
        self.assertAlmostEqual(scale, np.sqrt(26))
        np.testing.assert_allclose(z[:, -1], 1 / scale)
        np.testing.assert_allclose(z[:, :-1] * scale, x)
        self.assertLessEqual(np.max(np.linalg.norm(z, axis=1)), 1 + 1e-15)

    def test_all_represented_scanners_remain_in_normalized_training_risks(self):
        weights = self.helper("fold_weights")
        eligible = self.helper("eligible_scanners")
        site = np.repeat(np.arange(4), [6, 7, 8, 2])
        self.assertEqual(eligible(site, min_held_out=5, min_training=12), [0, 1, 2])
        w = weights(site, 0)
        self.assertAlmostEqual(w.sum(), 1)
        self.assertTrue(np.all(w[site == 0] == 0))
        self.assertTrue(np.all(w[site == 3] > 0))
        self.assertTrue(np.all(w[site != 0] == w[site != 0][0]))

    def test_logistic_gradient_hessian_and_full_penalty(self):
        stats = self.helper("logistic_stats")
        normalize = self.helper("prepare_features")
        data = self.data()
        z, _ = normalize(data["X"])
        w = np.ones(len(z)) / len(z)
        theta = np.array([0.2, -0.3, 0.1, 0.4])
        value, gradient, hessian, _, _ = stats(theta, z, data["y"], w, 0.1)
        step = 1e-5
        finite_gradient, finite_hessian = [], []
        for j in range(len(theta)):
            direction = np.eye(len(theta))[j] * step
            plus = stats(theta + direction, z, data["y"], w, 0.1)
            minus = stats(theta - direction, z, data["y"], w, 0.1)
            finite_gradient.append((plus[0] - minus[0]) / (2 * step))
            finite_hessian.append((plus[1] - minus[1]) / (2 * step))
        np.testing.assert_allclose(gradient, finite_gradient, atol=2e-10)
        np.testing.assert_allclose(hessian, np.asarray(finite_hessian).T, atol=2e-10)
        self.assertGreaterEqual(np.linalg.eigvalsh(hessian)[0], 0.1)
        unpenalized = stats(theta, z, data["y"], w, 0.0)[0]
        self.assertAlmostEqual(value - unpenalized, 0.05 * np.dot(theta, theta))

    def test_newton_fit_and_nested_transport_grids_match_endpoints(self):
        fit = self.helper("fit_logistic")
        trace = self.helper("trace_path")
        integrate = self.helper("integrated_change")
        data = self.data()
        z, _ = rv.prepare_features(data["X"])
        w0, w1 = rv.fold_weights(data["site"], 0), rv.fold_weights(data["site"], 1)
        endpoints = [fit(z, data["y"], w, 0.1) for w in (w0, w1)]
        path = trace(z, data["y"], w0, w1, 0.1, points=65)
        self.assertLessEqual(np.max(np.linalg.norm(path["gradients"], axis=1)), 1e-10)
        np.testing.assert_allclose(path["theta"][-1], endpoints[1]["theta"], atol=2e-9)
        actual = endpoints[1]["losses"] - endpoints[0]["losses"]
        errors = []
        for points in (17, 33, 65):
            estimate = integrate(path["psi"], points)
            pair_error = (estimate - actual)[0] - (estimate - actual)[1]
            errors.append(abs(pair_error))
        self.assertLess(errors[-1], 1e-8)
        self.assertLess(errors[-1], errors[0])

    def test_global_remainder_and_stationarity_error_bounds_are_conservative_on_toy_pairs(self):
        constants = self.helper("pair_constants")
        bounds = self.helper("global_path_bounds")
        data = self.data()
        z, _ = rv.prepare_features(data["X"])
        w0, w1 = rv.fold_weights(data["site"], 0), rv.fold_weights(data["site"], 1)
        c, j = constants(z, data["y"], np.array([0, 1, 2]), np.array([3, 4, 5]))
        b = bounds(z, w0, w1, 0.1, c, j)
        path = rv.trace_path(z, data["y"], w0, w1, 0.1)
        loss0 = np.logaddexp(0, -path["margins"][0])
        loss1 = np.logaddexp(0, -path["margins"][-1])
        actual, prediction = loss1 - loss0, -path["psi"][0]
        remainder = (actual - prediction)[:3] - (actual - prediction)[3:6]
        self.assertTrue(np.all(np.abs(remainder) <= b["remainder"]))
        self.assertTrue(np.all(b["prediction_lipschitz"] >= 0))

    def test_uncertain_source_sign_is_never_certified(self):
        classify = self.helper("classify_intervals")
        status = classify(np.array([1e-16, 1.0, -1.0]), np.array([1.0, 0.0, 2.0]),
                          np.zeros(3), np.array([2e-16, 0.0, 0.0]))
        np.testing.assert_array_equal(status, [0, 1, -1])

    def test_secondary_auc_handles_ties_and_constant_errors(self):
        auc = self.helper("failure_auc")
        self.assertEqual(auc(np.array([0.0, 1.0, 1.0]), np.array([0, 1, 0])), 0.75)
        self.assertIsNone(auc(np.array([0.0, 1.0]), np.array([0, 0])))

    def test_conditional_auc_retains_minimum_eight_cell_rule(self):
        conditional = self.helper("conditional_failure_auc")
        self.assertEqual(conditional(np.arange(7), np.array([0, 0, 0, 0, 1, 1, 1])),
                         (None, "below_minimum_n"))
        self.assertEqual(conditional(np.arange(8), np.array([0, 0, 0, 0, 1, 1, 1, 1])),
                         (1.0, "defined"))
        self.assertEqual(conditional(np.arange(8), np.zeros(8)), (None, "constant_error"))
        self.assertEqual(conditional(np.array([]), np.array([])), (None, "empty"))

    def test_complete_synthetic_pipeline_saves_every_setting_and_keeps_arrays_private(self):
        run = self.helper("run_experiment")
        with tempfile.TemporaryDirectory(prefix="synthetic_test_") as temp:
            output = Path(temp) / "result"
            summary = run(self.data(), output, lambdas=(0.1, 1.0), grids=(17, 33, 65),
                          min_held_out=5, min_training=20, progress=False)
            self.assertEqual(summary["settings"]["lambdas"], [0.1, 1.0])
            self.assertEqual(summary["settings"]["quadrature_points"], [17, 33, 65])
            self.assertEqual(summary["path_count"], 12)
            self.assertEqual(summary["observed_false_bound_checks"], 0)
            self.assertTrue((output / "private" / "lambda_0p1" / "endpoint_fits.npz").is_file())
            self.assertFalse(list(output.glob("*.npz")))
            self.assertTrue((output / "group_results.csv").is_file())
            self.assertIn("not rigorous", summary["interpretation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
