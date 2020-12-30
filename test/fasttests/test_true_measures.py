from qmcpy import *
from qmcpy.util import *
from numpy import *
import sys
import unittest

        
class TestUniform(unittest.TestCase):
    """ Unit tests for Uniform Measure. """

    def test_methods(self):
        s = Sobol(2)
        u = Uniform(2, lower_bound=-2, upper_bound=2)
        x = s.gen_samples(2**2)
        t = u.transform(x)
        self.assertTrue(t.shape==(4,2))
        j = u.jacobian(x)
        w = u.weight(x)
        u.set_dimension(3)
        s.set_dimension(3)
        self.assertTrue((u.a==tile(-2,3)).all() and (u.b==tile(2,3)).all())
        t2 = u.transform(s.gen_samples(2**2))
        self.assertTrue(t2.shape==(4,3))

    def test_errors(self):
        u = Uniform(2, lower_bound=[0,-2],upper_bound=[2,2])
        self.assertRaises(DimensionError,u.set_dimension,3)
        u = Uniform(2, lower_bound=[-2,-2],upper_bound=[5,2])
        self.assertRaises(DimensionError,u.set_dimension,3) 


class TestGaussian(unittest.TestCase):
    """ Unit tests for Gaussian Measure. """

    def test_methods(self):
        for decomp_type in ['Cholesky','PCA']:
            s = Sobol(2)
            g = Gaussian(2, mean=5, covariance=2)
            x = s.gen_samples(2**2)
            t = g.transform(x)
            self.assertTrue(t.shape==(4,2))
            j = g.jacobian(x)
            w = g.weight(x)
            g.set_dimension(3)
            s.set_dimension(3)
            self.assertTrue((g.mu==tile(5,3)).all() and (g.sigma==(2*eye(3))).all())
            t2 = g.transform(s.gen_samples(2**2))
            self.assertTrue(t2.shape==(4,3))

    def test_errors(self):
        g = Gaussian(2, mean=[1,1],covariance=[1,2])
        self.assertRaises(DimensionError,g.set_dimension,3)
        g = Gaussian(2, mean=[1,2],covariance=[2,2])
        self.assertRaises(DimensionError,g.set_dimension,3)


class TestBrownianMontion(unittest.TestCase):
    """ Unit tests for Brownian Motion Measure. """
    
    def test_set_dimension(self):
        bm = BrownianMotion(4, t_final=1, drift=0)
        self.assertTrue((bm.time_vec==array([1/4,1/2,3/4,1])).all())
        bm = BrownianMotion(4, t_final=2, drift=0)
        self.assertTrue((bm.time_vec==array([1/2,1,3/2,2])).all())
        bm = BrownianMotion(4, t_final=2, drift=1)
        self.assertTrue((bm.time_vec==bm.drift_time_vec).all())
        bm.set_dimension(5)
        self.assertTrue((bm.time_vec==bm.drift_time_vec).all())


class TestLebesgue(unittest.TestCase):
    """ Unit tests for Lebesgue Measure. """

    def test_transformers(self):
        for tf in [Uniform(3),Gaussian(3),BrownianMotion(3)]:
            l = Lebesgue(tf)
            l.set_dimension(2)
            tf.set_dimension(4)
            l.set_transform(tf)

if __name__ == "__main__":
    unittest.main()
