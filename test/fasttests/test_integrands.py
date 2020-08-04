""" Unit tests for subclasses of Integrands in QMCPy """

from qmcpy import *
from qmcpy.util import *
import sys
vinvo = sys.version_info
if vinvo[0]==3: import unittest
else: import unittest2 as unittest


class TestAsianOption(unittest.TestCase):
    """ Unit tests for AsianOption Integrand. """

    def test_f(self):
        distribution = Sobol(dimension=2)
        measure = BrownianMotion(distribution)
        integrand = AsianOption(measure)
        samples = integrand.measure.distribution.gen_samples(4)
        y = integrand.f(samples).squeeze()
        self.assertTrue(y.shape==(4,))        

    def test__dim_at_level(self):
        distribution = Sobol(dimension=4)
        measure = BrownianMotion(distribution)
        integrand = AsianOption(measure, multi_level_dimensions=[4,8])
        self.assertTrue(integrand._dim_at_level(0)==4)
        self.assertTrue(integrand._dim_at_level(1)==8)


class TestEuropeanOption(unittest.TestCase):
    """ Unit test for EuropeanOption Integrand. """

    def test_f(self):
        for option_type in ['call','put']:
            distribution = Sobol(dimension=3)
            measure = BrownianMotion(distribution)
            integrand = EuropeanOption(measure,call_put=option_type)
            samples = integrand.measure.distribution.gen_samples(4)
            y = integrand.f(samples.squeeze())
            self.assertTrue(y.shape==(4,))
            integrand.get_exact_value()


class TestKeister(unittest.TestCase):
    """ Unit tests for Keister Integrand. """

    def test_f(self):
        distribution = Sobol(dimension=3)
        measure = Uniform(distribution)
        integrand = Keister(measure)
        samples = integrand.measure.distribution.gen_samples(4)
        y = integrand.f(samples).squeeze()
        self.assertTrue(y.shape==(4,))


class TestLinear(unittest.TestCase):
    """ Unit tests for Linear Integrand. """

    def test_f(self):
        distribution = Sobol(dimension=3)
        measure = Uniform(distribution)
        integrand = Linear(measure)
        samples = integrand.measure.distribution.gen_samples(4)
        y = integrand.f(samples).squeeze()
        self.assertTrue(y.shape==(4,))


class TestCustomFun(unittest.TestCase):
    """ Unit tests for CustomFun Integrand. """

    def test_f(self):
        distribution = Sobol(dimension=3)
        measure = Uniform(distribution)
        integrand = CustomFun(measure, lambda x: x.sum(1))
        samples = integrand.measure.distribution.gen_samples(4)
        y = integrand.f(samples).squeeze()
        self.assertTrue(y.shape==(4,))


class TestCallOptions(unittest.TestCase):
    """ Unit tests for MLCallOptions Integrand. """

    def test_f(self):
        l = 3
        for option in ['European','Asian']:
            distribution = IIDStdGaussian()
            measure = Gaussian(distribution)
            integrand = MLCallOptions(measure,option=option)
            d = integrand._dim_at_level(l)
            integrand.measure.set_dimension(d)
            samples = integrand.measure.distribution.gen_samples(4)
            sums,cost = integrand.f(samples,l=l)
            self.assertTrue(sums.shape==(6,))

    def test__dim_at_level(self):
        l = 3
        # European 
        distribution = IIDStdGaussian()
        measure = Gaussian(distribution)
        integrand = MLCallOptions(measure,option='european')
        self.assertTrue(integrand._dim_at_level(0)==2**0)
        self.assertTrue(integrand._dim_at_level(3)==2**3)
        # Asian 
        distribution = IIDStdGaussian()
        measure = Gaussian(distribution)
        integrand = MLCallOptions(measure,option='asian')
        self.assertTrue(integrand._dim_at_level(0)==2**1)
        self.assertTrue(integrand._dim_at_level(3)==2**4)


if __name__ == "__main__":
    unittest.main()
