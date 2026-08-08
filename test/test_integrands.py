from qmcpy import *
from qmcpy.util import *
import numpy as np
import unittest
import scipy.stats


class TestIntegrand(unittest.TestCase):
    """General tests for Integrand"""

    def test_abstract_methods(self):
        n = 2**3
        d = 2
        integrands = [
            FinancialOption(
                DigitalNetB2(d, seed=7),
                option="ASIAN",
                call_put="call",
                asian_mean="arithmetic",
            ),
            FinancialOption(
                DigitalNetB2(d, seed=7),
                option="ASIAN",
                call_put="put",
                asian_mean="arithmetic",
            ),
            FinancialOption(
                DigitalNetB2(d, seed=7),
                option="ASIAN",
                call_put="call",
                asian_mean="geometric",
            ),
            FinancialOption(
                DigitalNetB2(d, seed=7),
                option="ASIAN",
                call_put="put",
                asian_mean="geometric",
            ),
            FinancialOption(
                DigitalNetB2(d, seed=7),
                option="ASIAN",
                call_put="put",
                asian_mean="geometric",
                level=1,
                d_coarsest=1,
            ),
            BoxIntegral(DigitalNetB2(d, seed=7), s=1),
            BoxIntegral(DigitalNetB2(d, seed=7), s=[3, 5, 7]),
            CustomFun(Uniform(DigitalNetB2(d, seed=7)), lambda x: x.prod(1)),
            CustomFun(
                Uniform(
                    Kumaraswamy(
                        SciPyWrapper(
                            DigitalNetB2(d, seed=7),
                            [scipy.stats.triang(c=0.1), scipy.stats.uniform()],
                        )
                    )
                ),
                lambda x: x.prod(1),
            ),
            CustomFun(
                Gaussian(DigitalNetB2(2, seed=7)),
                lambda x: np.moveaxis(x, -1, 0),
                dimension_indv=d,
            ),
            FinancialOption(
                DigitalNetB2(d, seed=7), option="EUROPEAN", call_put="call"
            ),
            FinancialOption(DigitalNetB2(d, seed=7), option="EUROPEAN", call_put="put"),
            Keister(DigitalNetB2(d, seed=7)),
            Keister(Gaussian(DigitalNetB2(d, seed=7))),
            Keister(BrownianMotion(Kumaraswamy(DigitalNetB2(d, seed=7)))),
            Linear0(DigitalNetB2(d, seed=7)),
        ]
        spawned_integrands = [integrand.spawn(levels=0)[0] for integrand in integrands]
        for integrand in integrands + spawned_integrands:
            x = integrand.discrete_distrib.gen_samples(n)
            s = str(integrand)
            for ptransform in ["None", "Baker", "C0", "C1", "C1sin", "C2sin", "C3sin"]:
                y = integrand.f(x, periodization_transform=ptransform)
                self.assertEqual(y.shape, (integrand.d_indv + (n,)))
                self.assertTrue(np.isfinite(y).all())
                self.assertEqual(y.dtype, np.float64)

    def test_keister(self, dims=3):
        k = Keister(DigitalNetB2(dims, seed=7))
        exact_integ = k.exact_integ(dims)
        x = k.discrete_distrib.gen_samples(2**10)
        y = k.f(x)
        self.assertAlmostEqual(y.mean(), exact_integ, places=2)


class TestIntegrandExamples(unittest.TestCase):
    """Focused tests for individual integrand implementations."""

    def _assert_scalar_finite_output(self, integrand, n=2**10):
        x = integrand.discrete_distrib.gen_samples(n)
        y = integrand.f(x)
        self.assertEqual(y.shape, (n,))
        self.assertTrue(np.isfinite(y).all())

    def _assert_genz_output(self, kind_func):
        for kind_coeff in [1, 2, 3]:
            integrand = Genz(
                DigitalNetB2(2, seed=7),
                kind_func=kind_func,
                kind_coeff=kind_coeff,
            )
            y = integrand(32)
            self.assertEqual(y.shape, (32,))

    def test_fourbranch2d(self):
        fb = FourBranch2d(DigitalNetB2(2, seed=7))
        self._assert_scalar_finite_output(fb)

    def test_multimodal2d(self):
        mm = Multimodal2d(DigitalNetB2(2, seed=7))
        self._assert_scalar_finite_output(mm)

    def test_genz_oscillatory_all_coeffs(self):
        self._assert_genz_output("OSCILLATORY")

    def test_genz_corner_peak_all_coeffs(self):
        self._assert_genz_output("CORNER PEAK")

    def test_genz_invalid_kind_func(self):
        self.assertRaises(ParameterError, Genz, DigitalNetB2(2, seed=7), kind_func="INVALID")

    def test_genz_invalid_kind_coeff(self):
        self.assertRaises(ParameterError, Genz, DigitalNetB2(2, seed=7), kind_coeff=4)

    def test_genz_spawn(self):
        ig = Genz(DigitalNetB2(2, seed=7), kind_func="CORNER PEAK", kind_coeff=2)
        spawned = ig.spawn(levels=0)
        self.assertEqual(len(spawned), 1)

    def test_sin1d_basic_and_spawn(self):
        ig = Sin1d(DigitalNetB2(1, seed=7), k=2)
        y = ig(64)
        self.assertEqual(y.shape, (64,))
        self.assertTrue(np.isfinite(y).all())
        spawned = ig.spawn(levels=0)
        self.assertEqual(len(spawned), 1)

    def test_ishigami_basic_and_dimension_error(self):
        ig = Ishigami(DigitalNetB2(3, seed=7), a=7, b=0.1)
        y = ig(64)
        self.assertEqual(y.shape, (64,))
        self.assertTrue(np.isfinite(y).all())
        self.assertRaises(ParameterError, Ishigami, DigitalNetB2(2, seed=7))

    def test_ishigami_exact_helpers(self):
        indices = np.array(
            [[True, False, False], [False, True, False], [False, False, True]],
            dtype=bool,
        )
        sens = Ishigami._exact_sensitivity_indices(indices, a=7, b=0.1)
        self.assertEqual(sens.shape, (2, 3))
        fu = Ishigami._exact_fu_functions(
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            [[], [0], [1, 2]],
            a=7,
            b=0.1,
        )
        self.assertEqual(fu.shape, (2, 3))

    def test_hartmann6d_smoke(self):
        try:
            import botorch  # noqa: F401
        except Exception:
            self.skipTest("botorch not installed")
        ig = Hartmann6d(DigitalNetB2(6, seed=7))
        y = ig(32)
        self.assertEqual(y.shape, (32,))
        self.assertTrue(np.isfinite(y).all())

    def test_financial_option_invalid_inputs(self):
        self.assertRaises(
            AssertionError,
            FinancialOption,
            DigitalNetB2(2, seed=7),
            option="ASIAN",
            call_put="BAD",
        )
        self.assertRaises(
            AssertionError,
            FinancialOption,
            DigitalNetB2(2, seed=7),
            option="ASIAN",
            asian_mean="BAD",
        )
        self.assertRaises(
            AssertionError,
            FinancialOption,
            DigitalNetB2(2, seed=7),
            option="ASIAN",
            asian_mean_quadrature_rule="BAD",
        )


class TestBayesianLRCoeffs(unittest.TestCase):
    """Tests for BayesianLRCoeffs methods not covered by unit tests."""

    def setUp(self):
        self.ig = BayesianLRCoeffs(
            DigitalNetB2(3, seed=7),
            feature_array=np.arange(8).reshape((4, 2)),
            response_vector=[0, 0, 1, 1],
        )

    def test_basic_evaluation(self):
        y = self.ig(64)
        # shape is (2, num_coeffs, n) = (2, 3, 64)
        self.assertEqual(y.shape, (2, 3, 64))

    def test_bound_fun(self):
        bound_low = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        bound_high = np.array([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
        low, high = self.ig.bound_fun(bound_low, bound_high)
        self.assertEqual(low.shape, (3,))
        self.assertEqual(high.shape, (3,))
        self.assertTrue((low <= high).all())

    def test_bound_fun_denominator_spans_zero(self):
        # den_bounds_low <= 0 <= den_bounds_high  → (-inf, +inf) bounds
        bound_low = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
        bound_high = np.array([[0.2, 0.3, 0.4], [0.1, 0.2, 0.3]])
        low, high = self.ig.bound_fun(bound_low, bound_high)
        self.assertTrue(np.all(low == -np.inf) or np.any(np.isinf(low)))

    def test_dependency(self):
        comb_flags = np.array([True, False, True])
        dep = self.ig.dependency(comb_flags)
        self.assertEqual(dep.shape, (2, 3))

    def test_invalid_dimension(self):
        # Dimension 2 but feature_array has 2 features → expects d=3
        self.assertRaises(
            ParameterError,
            BayesianLRCoeffs,
            DigitalNetB2(2, seed=7),
            np.arange(8).reshape((4, 2)),
            [0, 0, 1, 1],
        )

    def test_invalid_response_vector(self):
        # Response vector with non-binary values
        self.assertRaises(
            ParameterError,
            BayesianLRCoeffs,
            DigitalNetB2(3, seed=7),
            np.arange(8).reshape((4, 2)),
            [0, 0, 2, 1],
        )


if __name__ == "__main__":
    unittest.main()
