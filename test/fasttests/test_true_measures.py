from qmcpy import *
from qmcpy.util import *
from numpy import *
import sys
import unittest

        
class TestUniform(unittest.TestCase):
    """ Unit tests for Uniform Measure. """

    def test_methods(self):
        u = Uniform(Sobol(2), lower_bound=-2, upper_bound=2)
        t = u.gen_samples(2**2)
        self.assertTrue(t.shape==(4,2))
        self.assertTrue((u.a==tile(-2,2)).all() and (u.b==tile(2,2)).all())

    def test_errors(self):
        u = Uniform(Sobol(2), lower_bound=[0,-2],upper_bound=[2,2])
        self.assertRaises(DimensionError,u._set_dimension,3)
        u = Uniform(Sobol(2), lower_bound=[-2,-2],upper_bound=[5,2])
        self.assertRaises(DimensionError,u._set_dimension,3) 


class TestGaussian(unittest.TestCase):
    """ Unit tests for Gaussian Measure. """

    def test_methods(self):
        for decomp_type in ['Cholesky','PCA']:
            g = Gaussian(Sobol(2), mean=5, covariance=2)
            t = g.gen_samples(2**2)
            self.assertTrue(t.shape==(4,2))
            self.assertTrue((g.mu==tile(5,2)).all() and (g.sigma==(2*eye(2))).all())

    def test_errors(self):
        g = Gaussian(Sobol(2), mean=[1,1],covariance=[1,2])
        self.assertRaises(DimensionError,g._set_dimension,3)
        g = Gaussian(Sobol(2), mean=[1,2],covariance=[2,2])
        self.assertRaises(DimensionError,g._set_dimension,3)


class TestBrownianMontion(unittest.TestCase):
    """ Unit tests for Brownian Motion Measure. """
    
    def test_set_dimension(self):
        bm = BrownianMotion(Sobol(4), t_final=1, drift=0)
        self.assertTrue((bm.time_vec==array([1/4,1/2,3/4,1])).all())
        bm = BrownianMotion(Sobol(4), t_final=2, drift=0)
        self.assertTrue((bm.time_vec==array([1/2,1,3/2,2])).all())
        bm = BrownianMotion(Sobol(4), t_final=2, drift=1)
        self.assertTrue((bm.time_vec==bm.drift_time_vec).all())
        bm._set_dimension(5)
        self.assertTrue((bm.time_vec==bm.drift_time_vec).all())


class TestLebesgue(unittest.TestCase):
    """ Unit tests for Lebesgue Measure. """

    def test_transformers(self):
        for tf in [Uniform(Sobol(4)),Gaussian(Sobol(4)),BrownianMotion(Sobol(4))]:
            l = Lebesgue(tf)
            l._set_dimension_r(2)

if __name__ == "__main__":
    unittest.main()
