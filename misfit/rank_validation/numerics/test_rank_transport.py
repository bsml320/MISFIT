"""Synthetic numerical sanity checks only: no MRI data or CNN validation.

Run with Python + NumPy, without SciPy:
    python3 -m unittest -v test_rank_transport
"""

import importlib
import inspect
import unittest

import numpy as np

try:
    rt = importlib.import_module("misfit.rank_validation.numerics.rank_transport")
except ModuleNotFoundError as exc:
    if exc.name != "rank_transport":
        raise
    rt = None


class SyntheticChecks(unittest.TestCase):
    def helper(self, name):
        self.assertIsNotNone(rt, "The rank_transport helper module is not implemented")
        function = getattr(rt, name, None)
        self.assertTrue(callable(function), f"The {name} helper is not implemented")
        return function

    @staticmethod
    def matrix_quadratic():
        h0 = np.array([[2.0, 0.4], [0.4, 1.5]])
        h1 = np.array([[3.0, -0.2], [-0.2, 2.2]])
        b0, b1 = np.array([0.8, -0.4]), np.array([-0.6, 0.9])
        q = np.array([[0.6, 0.2], [0.2, -0.3]])
        v = np.array([0.7, -0.8])
        theta = lambda t: np.linalg.solve((1 - t) * h0 + t * h1, (1 - t) * b0 + t * b1)
        hessian = lambda t, x: (1 - t) * h0 + t * h1
        grad_delta = lambda t, x: (h1 - h0) @ x - (b1 - b0)
        grad_gap = lambda x: q @ x + v
        gap = lambda x: 0.5 * x @ q @ x + v @ x + 0.2
        # Endpoints are solved directly from the two objectives, independently
        # of the transport helper and its path integration.
        exact = gap(np.linalg.solve(h1, b1)) - gap(np.linalg.solve(h0, b0))
        return theta, grad_gap, hessian, grad_delta, exact

    @staticmethod
    def scalar_quadratic(h0=2.0, h1=3.0, b0=1.0, b1=4.0, a=1.0):
        theta = lambda t: np.array([(b0 + t * (b1 - b0)) / (h0 + t * (h1 - h0))])
        grad_gap = lambda x: np.array([a])
        hessian = lambda t, x: np.array([[h0 + t * (h1 - h0)]])
        grad_delta = lambda t, x: np.array([(h1 - h0) * x[0] - (b1 - b0)])
        c = (h1 - h0) * b0 - (b1 - b0) * h0
        # Psi(t)=a*C/h(t)^2. Thus this K is a global analytic bound,
        # obtained from h(t)>=min(h0,h1), never from sampled derivatives.
        k_bound = 2 * abs(a * c * (h1 - h0)) / min(h0, h1) ** 3
        exact = a * (b1 / h1 - b0 / h0)
        return theta, grad_gap, hessian, grad_delta, exact, k_bound

    def test_exact_matrix_quadratic_identity_against_analytic_endpoints(self):
        integrate = self.helper("integrate_transport")
        theta, dg, h, dr, exact = self.matrix_quadratic()
        transported = integrate(theta, dg, h, dr, panels=256)
        self.assertAlmostEqual(transported, exact, delta=2e-10)

    def test_composite_simpson_quadrature_converges(self):
        integrate = self.helper("integrate_transport")
        theta, dg, h, dr, exact = self.scalar_quadratic()[:5]
        errors = [abs(integrate(theta, dg, h, dr, panels=n) - exact) for n in (4, 8, 16, 32, 64)]
        self.assertTrue(all(next_error < error / 12 for error, next_error in zip(errors, errors[1:])), errors)
        self.assertLess(errors[-1], 1e-8)

    def test_transport_sign_and_first_order_value(self):
        integrand = self.helper("transport_integrand")
        first_order = self.helper("first_order_change")
        theta, dg, h, dr, exact, _ = self.scalar_quadratic()
        self.assertAlmostEqual(integrand(0.0, theta, dg, h, dr), -1.25)
        self.assertAlmostEqual(first_order(theta, dg, h, dr), 1.25)
        self.assertGreater(exact, 0)

    def test_first_order_remainder_is_below_analytic_global_bound(self):
        first_order = self.helper("first_order_change")
        theta, dg, h, dr, exact, k_bound = self.scalar_quadratic()
        remainder = abs(exact - first_order(theta, dg, h, dr))
        self.assertAlmostEqual(remainder, 5 / 12)
        self.assertAlmostEqual(k_bound / 2, 5 / 8)
        self.assertLess(remainder, k_bound / 2)

    def test_zero_remainder_for_constant_hessian_linear_gap(self):
        first_order = self.helper("first_order_change")
        theta, dg, h, dr, exact, k_bound = self.scalar_quadratic(h1=2)
        self.assertEqual(k_bound, 0)
        self.assertAlmostEqual(first_order(theta, dg, h, dr), exact)

    def test_common_theta_dependent_loss_shift_cancels_in_pair_transport(self):
        integrate = self.helper("integrate_transport")
        theta, _, h, dr, _ = self.matrix_quadratic()
        loss_i = lambda x: 0.5 * np.dot(x - [1.0, -0.5], x - [1.0, -0.5])
        loss_j = lambda x: 0.5 * np.dot(x - [-0.3, 0.2], x - [-0.3, 0.2])
        grad_i = lambda x: x - np.array([1.0, -0.5])
        grad_j = lambda x: x - np.array([-0.3, 0.2])
        grad_gap = lambda x: grad_i(x) - grad_j(x)
        transported = integrate(theta, grad_gap, h, dr, panels=256)
        endpoint_changes = []
        common = lambda x: 5000 * x[0] ** 2 + 1200 * x[1] + 7.3
        grad_common = lambda x: np.array([10000 * x[0], 1200.0])
        self.assertGreater(abs(common(theta(1)) - common(theta(0))), 1)
        for scale in (0.0, 1.0):
            gap0 = (loss_i(theta(0)) + scale * common(theta(0))) - (loss_j(theta(0)) + scale * common(theta(0)))
            gap1 = (loss_i(theta(1)) + scale * common(theta(1))) - (loss_j(theta(1)) + scale * common(theta(1)))
            shifted_grad_gap = lambda x: ((grad_i(x) + scale * grad_common(x))
                                          - (grad_j(x) + scale * grad_common(x)))
            shifted_transport = integrate(theta, shifted_grad_gap, h, dr, panels=256)
            endpoint_changes.append(gap1 - gap0)
            self.assertAlmostEqual(transported, gap1 - gap0, delta=2e-10)
            self.assertAlmostEqual(shifted_transport, transported, delta=2e-12)
        self.assertAlmostEqual(*endpoint_changes)

    @staticmethod
    def containing_fold_parameters():
        # Three shared containing folds for the same pair, each with its own
        # strongly convex source objective, ending at one shared target (3,4).
        # This is not an MRI fold simulation.
        return ((2.0, 3.0, 1.0, 4.0), (3.0, 3.0, 1.8, 4.0), (2.5, 3.0, 1.5, 4.0))

    def test_shared_containing_fold_average_identity_and_rank_bound(self):
        integrate = self.helper("integrate_transport")
        first_order = self.helper("first_order_change")
        certify = self.helper("certify_pair")
        targets = {(h1, b1) for _, h1, _, b1 in self.containing_fold_parameters()}
        self.assertEqual(len(targets), 1, "All containing folds must converge to one shared target objective")
        transports, predictions, remainders, initials, finals = [], [], [], [], []
        for h0, h1, b0, b1 in self.containing_fold_parameters():
            theta, dg, h, dr, exact, k_bound = self.scalar_quadratic(h0, h1, b0, b1)
            transports.append(integrate(theta, dg, h, dr, panels=256))
            predictions.append(first_order(theta, dg, h, dr))
            remainders.append(k_bound / 2)
            initials.append(b0 / h0)
            finals.append(b1 / h1)
        exact_average_change = np.mean(finals) - np.mean(initials)
        self.assertAlmostEqual(np.mean(transports), exact_average_change, delta=1e-10)
        self.assertLessEqual(abs(exact_average_change - np.mean(predictions)), np.mean(remainders))
        for offset, status in ((-0.8, "inverted"), (2.0, "preserved")):
            result = certify(np.mean(initials) + offset, np.mean(predictions), np.mean(remainders))
            self.assertEqual(result.status, status)
            self.assertLessEqual(result.lower, np.mean(finals) + offset)
            self.assertGreaterEqual(result.upper, np.mean(finals) + offset)

    def test_report_states_synthetic_scope_and_exposes_observed_errors(self):
        try:
            report_module = importlib.import_module("misfit.rank_validation.numerics.report_checks")
        except ModuleNotFoundError as exc:
            if exc.name != "report_checks":
                raise
            report_module = None
        self.assertIsNotNone(report_module, "The reproducible numerical report is not implemented")
        report = report_module.build_report()
        self.assertIn("not MRI", report["scope"])
        self.assertIn("not CNN", report["scope"])
        self.assertLess(report["matrix_quadratic"]["absolute_identity_error"], 2e-10)
        self.assertEqual(report["shared_containing_folds"]["fold_count"], 3)
        self.assertEqual(report["shared_containing_folds"]["certificate"], "inverted")
        self.assertIn("transport_derivative", report["ridge_logistic"])
        self.assertGreater(report["scalar_quadratic"]["analytic_remainder_bound"],
                           report["scalar_quadratic"]["absolute_first_order_remainder"])

    def test_certificates_cover_preservation_and_inversion_in_both_directions(self):
        certify = self.helper("certify_pair")
        # Analytic scalar example: delta=5/6, first order=5/4, K/2=5/8.
        for a, offset, expected in ((1, 2, "preserved"), (1, -4, "preserved"),
                                    (1, -0.8, "inverted"), (-1, 0.8, "inverted")):
            with self.subTest(a=a, offset=offset):
                initial_gap = a * 0.5 + offset
                result = certify(initial_gap, a * 1.25, 0.625)
                self.assertEqual(result.status, expected)
                actual_endpoint = a * (4 / 3) + offset
                self.assertLessEqual(result.lower, actual_endpoint)
                self.assertGreaterEqual(result.upper, actual_endpoint)

    def test_ties_and_intervals_touching_zero_are_not_strictly_certified(self):
        certify = self.helper("certify_pair")
        for args in ((0.0, 2.0, 0.0), (1.0, -1.0, 0.0),
                     (1.0, 0.0, 1.0), (1.0, -2.0, 1.0), (-1.0, 1.0, 0.0)):
            with self.subTest(args=args):
                self.assertEqual(certify(*args).status, "undetermined")

    def test_uncertain_initial_gap_sign_prevents_false_preservation_certificate(self):
        certify = self.helper("certify_pair")
        self.assertIn("initial_gap_error", inspect.signature(certify).parameters)
        # A true source gap of -1e-16 is compatible with this +1e-16 estimate.
        # Although the target gap is certainly positive, source order is unknown.
        result = certify(1e-16, 1.0, 0.0, initial_gap_error=2e-16)
        self.assertEqual(result.status, "undetermined")
        self.assertGreater(result.lower, 0)
        self.assertLessEqual(result.lower, 1.0 - 1e-16)
        self.assertGreaterEqual(result.upper, 1.0 - 1e-16)

    def test_initial_gap_uncertainty_propagates_to_endpoint_and_preserves_known_sign(self):
        certify = self.helper("certify_pair")
        self.assertIn("initial_gap_error", inspect.signature(certify).parameters)
        result = certify(1.0, 0.0, 0.0, initial_gap_error=0.25)
        self.assertEqual(result.status, "preserved")
        self.assertLessEqual(result.lower, 0.75)
        self.assertGreaterEqual(result.upper, 1.25)
        self.assertEqual(certify(-0.1, 1.0, 0.1, initial_gap_error=0.01).status, "inverted")
        for invalid_error in (-1.0, np.nan, np.inf, None):
            with self.subTest(initial_gap_error=invalid_error):
                with self.assertRaises(ValueError):
                    certify(1.0, 0.0, 0.0, initial_gap_error=invalid_error)

    def test_valid_analytic_bounds_never_produce_false_certificate_on_parameter_grid(self):
        certify = self.helper("certify_pair")
        first_order = self.helper("first_order_change")
        counts = {"preserved": 0, "inverted": 0, "undetermined": 0}
        for h0 in (0.8, 2.0):
            for h1 in (1.2, 3.0):
                for b0 in (-1.0, 1.0):
                    for b1 in (-2.0, 4.0):
                        for a in (-1.0, 1.0):
                            path = self.scalar_quadratic(h0, h1, b0, b1, a)
                            theta, dg, h, dr, exact, k_bound = path
                            prediction = first_order(theta, dg, h, dr)
                            for offset in (-8.0, -0.8, 0.0, 0.8, 8.0):
                                initial_gap = a * b0 / h0 + offset
                                final_gap = a * b1 / h1 + offset
                                result = certify(initial_gap, prediction, k_bound / 2)
                                counts[result.status] += 1
                                self.assertLessEqual(result.lower - 1e-13, final_gap)
                                self.assertGreaterEqual(result.upper + 1e-13, final_gap)
                                if result.status == "preserved":
                                    self.assertGreater(initial_gap * final_gap, 0)
                                elif result.status == "inverted":
                                    self.assertLess(initial_gap * final_gap, 0)
        self.assertTrue(all(count > 0 for count in counts.values()), counts)

    def test_singular_or_non_positive_hessian_is_rejected(self):
        integrand = self.helper("transport_integrand")
        theta, dg, _, dr, _ = self.matrix_quadratic()
        for matrix in (np.zeros((2, 2)), np.diag([1.0, -1.0]), np.array([[1.0, 0.3], [0.0, 1.0]])):
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    integrand(0.0, theta, dg, lambda t, x: matrix, dr)

    def test_nonfinite_values_bad_dimensions_and_quadrature_inputs_are_rejected(self):
        integrate = self.helper("integrate_transport")
        integrand = self.helper("transport_integrand")
        theta, dg, h, dr, _ = self.matrix_quadratic()
        for panels in (0, 1, 3, 4.0, True):
            with self.subTest(panels=panels):
                with self.assertRaises(ValueError):
                    integrate(theta, dg, h, dr, panels=panels)
        for t in (-0.1, 1.1, np.nan):
            with self.subTest(t=t):
                with self.assertRaises(ValueError):
                    integrand(t, theta, dg, h, dr)
        for bad_gradient in (lambda x: np.array([1.0]), lambda x: np.array([np.nan, 1.0])):
            with self.assertRaises(ValueError):
                integrand(0.0, theta, bad_gradient, h, dr)

    def test_certificate_rejects_invalid_or_missing_bounds(self):
        certify = self.helper("certify_pair")
        for args in ((1, 1, -1), (1, 1, np.inf), (np.nan, 1, 1),
                     (1, np.nan, 1), (1, 1, None), (1e308, 1e308, 1)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    certify(*args)

    @staticmethod
    def ridge_logistic():
        x0 = np.array([[1, 0], [0, 1], [1, 1], [-1, 0.5], [0.5, -1.0]], dtype=float)
        y0 = np.array([1, 0, 1, 0, 1], dtype=float)
        x1 = np.array([[1, -0.5], [0.2, 1], [-1, 1], [1, 1]], dtype=float)
        y1 = np.array([0, 1, 1, 0], dtype=float)
        ridge = 0.8

        def parts(x, y, theta):
            z = x @ theta
            p = np.exp(-np.logaddexp(0.0, -z))
            value = np.mean(np.logaddexp(0.0, z) - y * z) + ridge * np.dot(theta, theta) / 2
            gradient = x.T @ (p - y) / len(y) + ridge * theta
            hessian = (x.T * (p * (1 - p))) @ x / len(y) + ridge * np.eye(2)
            return value, gradient, hessian

        def objective(t, theta):
            left, right = parts(x0, y0, theta), parts(x1, y1, theta)
            return tuple((1 - t) * a + t * b for a, b in zip(left, right))

        def theta_at(t):
            theta = np.zeros(2)
            for _ in range(30):
                _, gradient, hessian = objective(t, theta)
                if np.linalg.norm(gradient, ord=np.inf) < 2e-14:
                    return theta
                theta = theta - np.linalg.solve(hessian, gradient)
            raise AssertionError("Synthetic ridge-logistic Newton solve did not converge")

        hessian_at = lambda t, theta: objective(t, theta)[2]
        delta_gradient = lambda t, theta: parts(x1, y1, theta)[1] - parts(x0, y0, theta)[1]
        u, v = np.array([0.8, -0.4]), np.array([-0.3, 1.1])
        gap = lambda theta: np.logaddexp(0.0, -u @ theta) - np.logaddexp(0.0, v @ theta)
        grad_gap = lambda theta: -u * np.exp(-np.logaddexp(0.0, u @ theta)) - v * np.exp(-np.logaddexp(0.0, -v @ theta))

        def derivative_terms(t):
            """Analytic Psi' terms, independent of finite differences of Psi."""
            theta = theta_at(t)
            hessian = hessian_at(t, theta)
            c = grad_gap(theta)
            direction = np.linalg.solve(hessian, delta_gradient(t, theta))
            p_u = np.exp(-np.logaddexp(0.0, u @ theta))
            p_v = np.exp(-np.logaddexp(0.0, -v @ theta))
            gap_hessian = (np.outer(u, u) * p_u * (1 - p_u)
                           - np.outer(v, v) * p_v * (1 - p_v))
            delta_hessian = parts(x1, y1, theta)[2] - parts(x0, y0, theta)[2]

            def third_contraction(x):
                # T[direction] = mean p(1-p)(1-2p)(x.direction) x x^T.
                # The ridge penalty contributes no third derivative.
                p = np.exp(-np.logaddexp(0.0, -(x @ theta)))
                coefficient = p * (1 - p) * (1 - 2 * p) * (x @ direction)
                return (x.T * coefficient) @ x / len(x)

            contracted_third = (1 - t) * third_contraction(x0) + t * third_contraction(x1)
            return {
                "gap_curvature": float(-direction @ gap_hessian @ direction),
                "objective_shift_curvature": float(-2 * c @ np.linalg.solve(hessian, delta_hessian @ direction)),
                "objective_third_derivative": float(c @ np.linalg.solve(hessian, contracted_third @ direction)),
                "third_contraction_norm": float(np.linalg.norm(contracted_third, ord=2)),
            }

        return theta_at, grad_gap, hessian_at, delta_gradient, gap, objective, ridge, derivative_terms

    def test_smooth_ridge_logistic_transport_matches_independent_endpoint_losses(self):
        integrate = self.helper("integrate_transport")
        theta, dg, h, dr, gap, objective, ridge, _ = self.ridge_logistic()
        for t in (0.0, 0.2, 0.7, 1.0):
            solution = theta(t)
            self.assertLess(np.linalg.norm(objective(t, solution)[1], ord=np.inf), 2e-14)
            self.assertGreaterEqual(np.linalg.eigvalsh(h(t, solution))[0], ridge)
        exact_endpoint_difference = gap(theta(1)) - gap(theta(0))
        transported = integrate(theta, dg, h, dr, panels=64)
        self.assertAlmostEqual(transported, exact_endpoint_difference, delta=1e-9)

    def test_full_transport_derivative_with_nonzero_logistic_third_derivative(self):
        integrand = self.helper("transport_integrand")
        path = self.ridge_logistic()
        self.assertEqual(len(path), 8, "The analytic logistic derivative fixture is not implemented")
        theta, dg, h, dr, _, _, _, derivative_terms = path
        t = 0.37
        terms = derivative_terms(t)
        component_names = ("gap_curvature", "objective_shift_curvature", "objective_third_derivative")
        self.assertTrue(all(abs(terms[name]) > 1e-10 for name in component_names), terms)
        self.assertGreater(terms["third_contraction_norm"], 1e-8)
        analytic_derivative = sum(terms[name] for name in component_names)
        errors = []
        for step in (2e-3, 1e-3):
            finite_difference = (integrand(t + step, theta, dg, h, dr)
                                 - integrand(t - step, theta, dg, h, dr)) / (2 * step)
            errors.append(abs(finite_difference - analytic_derivative))
        self.assertLess(errors[1], 1e-8)
        self.assertLess(errors[1], 0.35 * errors[0], errors)


if __name__ == "__main__":
    unittest.main(verbosity=2)
