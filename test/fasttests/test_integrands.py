from qmcpy import *
from qmcpy.util import *
import unittest


class TestAsianOption(unittest.TestCase):
    """ Unit tests for AsianOption Integrand. """

    def test_fs(self):
        ao = AsianOption(Sobol(2))
        x = ao.discrete_distrib.gen_samples(2**2)
        y = ao.f(x)
        self.assertTrue(y.shape==(4,1))
        for ptransform in ['Baker','C0','C1','C1sin','C2sin','C3sin','None']:
            yp = ao.f_periodized(x,ptransform=ptransform)
            self.assertTrue(yp.shape==(4,1))

    def test__dim_at_level(self):
        ao = AsianOption(Sobol(), multi_level_dimensions=[4,8])
        self.assertTrue(ao._dim_at_level(0)==4)
        self.assertTrue(ao._dim_at_level(1)==8)


class TestEuropeanOption(unittest.TestCase):
    """ Unit test for EuropeanOption Integrand. """

    def test_fs(self):
        for option_type in ['call','put']:
            eo = EuropeanOption(Sobol(2),call_put=option_type)
            x = eo.discrete_distrib.gen_samples(4)
            y = eo.f(x)
            self.assertTrue(y.shape==(4,1))
            eo.get_exact_value()


class TestKeister(unittest.TestCase):
    """ Unit tests for Keister Integrand. """

    def test_f(self):
        k = Keister(Gaussian(Lattice(2),mean=1,covariance=3))
        x = k.discrete_distrib.gen_samples(2**2)
        y = k.f(x)
        self.assertTrue(y.shape==(4,1))
        y2 = k.f(x)
        self.assertTrue(y2.shape==(4,1))


class TestLinear(unittest.TestCase):
    """ Unit tests for Linear Integrand. """

    def test_f(self):
        l = Linear0(Halton(3))
        y = l.f_periodized(l.discrete_distrib.gen_samples(2**2),'c1sin')
        self.assertTrue(y.shape==(4,1))


class TestCustomFun(unittest.TestCase):
    """ Unit tests for CustomFun Integrand. """

    def test_f(self):
        cf = CustomFun(Uniform(Lattice(2)), lambda x: x.sum(1))
        y = cf.f_periodized(cf.discrete_distrib.gen_samples(2**2))
        self.assertTrue(y.shape==(4,1))


class TestCallOptions(unittest.TestCase):
    """ Unit tests for MLCallOptions Integrand. """

    def test_f(self):
        l = 3
        for option in ['European','Asian']:
            mlco = MLCallOptions(IIDStdUniform(),option=option)
            d = mlco._dim_at_level(l)
            true_d = 2**3 if option=='European' else 2**4
            self.assertTrue(d==true_d)
            mlco.true_measure._set_dimension_r(d)
            y = mlco.f_periodized(mlco.discrete_distrib.gen_samples(6),'c3sin',l=l)
            self.assertTrue(y.shape==(6,1))


if __name__ == "__main__":
    unittest.main()
