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

    def test_FourBranch2d(self):
        fb = FourBranch2d(DigitalNetB2(2, seed=7))
        x = fb.discrete_distrib.gen_samples(2**10)
        y = fb.f(x)
        self.assertEqual(y.shape, (2**10,))
        self.assertTrue(np.isfinite(y).all())
    
    def test_multimodal2d(self):
        mm = Multimodal2d(DigitalNetB2(2, seed=7))
        x = mm.discrete_distrib.gen_samples(2**10)
        y = mm.f(x)
        self.assertEqual(y.shape, (2**10,))
        self.assertTrue(np.isfinite(y).all())

    def test_genz_oscillatory_all_coeffs(self):
        for kind_coeff in [1, 2, 3]:
            ig = Genz(DigitalNetB2(2, seed=7), kind_func="OSCILLATORY", kind_coeff=kind_coeff)
            y = ig(32)
            self.assertEqual(y.shape, (32,))

    def test_genz_corner_peak_all_coeffs(self):
        for kind_coeff in [1, 2, 3]:
            ig = Genz(DigitalNetB2(2, seed=7), kind_func="CORNER PEAK", kind_coeff=kind_coeff)
            y = ig(32)
            self.assertEqual(y.shape, (32,))

    def test_genz_invalid_kind_func(self):
        self.assertRaises(ParameterError, Genz, DigitalNetB2(2, seed=7), kind_func="INVALID")

    def test_genz_invalid_kind_coeff(self):
        self.assertRaises(ParameterError, Genz, DigitalNetB2(2, seed=7), kind_coeff=4)

    def test_genz_spawn(self):
        ig = Genz(DigitalNetB2(2, seed=7), kind_func="CORNER PEAK", kind_coeff=2)
        spawned = ig.spawn(levels=0)
        self.assertEqual(len(spawned), 1)

if __name__ == "__main__":
    unittest.main()
