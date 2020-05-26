""" Unit tests for subclasses of Integrands in QMCPy """

from qmcpy import *
from qmcpy.util import *
from numpy import arange
import unittest


class TestAsianCall(unittest.TestCase):
    """ Unit tests for AsianCall Integrand. """

    def test_f(self):
        distribution = Sobol(dimension=2)
        measure = BrownianMotion(distribution)
        integrand = AsianCall(measure)
        samples = integrand.measure.distribution.gen_samples(4)
        y = integrand.f(samples).squeeze()
        self.assertTrue(y.shape==(4,))        

    def test_dim_at_level(self):
        distribution = Sobol(dimension=4)
        measure = BrownianMotion(distribution)
        integrand = AsianCall(measure, multi_level_dimensions=[4,8])
        self.assertTrue(integrand.dim_at_level(0)==4)
        self.assertTrue(integrand.dim_at_level(1)==8)


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
            integrand.get_fair_price()


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


class TestQuickConstruct(unittest.TestCase):
    """ Unit tests for QuickConstruct Integrand. """

    def test_f(self):
        distribution = Sobol(dimension=3)
        measure = Uniform(distribution)
        integrand = QuickConstruct(measure, lambda x: x.sum(1))
        samples = integrand.measure.distribution.gen_samples(4)
        y = integrand.f(samples).squeeze()
        self.assertTrue(y.shape==(4,))


class TestCallOptions(unittest.TestCase):
    """ Unit tests for MLMCCallOptions Integrand. """

    def test_f(self):
        l = 3
        for option in ['European','Asian']:
            distribution = IIDStdGaussian()
            measure = Gaussian(distribution)
            integrand = MLMCCallOptions(measure,option=option)
            d = integrand.dim_at_level(l)
            integrand.measure.set_dimension(d)
            samples = integrand.measure.distribution.gen_samples(4)
            sums,cost = integrand.f(samples,l=l)
            self.assertTrue(sums.shape==(6,))

    def test_dim_at_level(self):
        l = 3
        # European 
        distribution = IIDStdGaussian()
        measure = Gaussian(distribution)
        integrand = MLMCCallOptions(measure,option='european')
        self.assertTrue(integrand.dim_at_level(0)==2**0)
        self.assertTrue(integrand.dim_at_level(3)==2**3)
        # Asian 
        distribution = IIDStdGaussian()
        measure = Gaussian(distribution)
        integrand = MLMCCallOptions(measure,option='asian')
        self.assertTrue(integrand.dim_at_level(0)==2**1)
        self.assertTrue(integrand.dim_at_level(3)==2**4)


if __name__ == "__main__":
    unittest.main()
