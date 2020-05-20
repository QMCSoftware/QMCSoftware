""" Unit tests for subclasses of TrueMeasures in QMCPy """

from qmcpy import *
from qmcpy.util import *
from numpy import *
import unittest

        
class TestUniform(unittest.TestCase):
    """ Unit tests for Uniform Measure. """

    def test_gen_mimic_samples(self):
        distribution = Sobol(dimension=3)
        measure = Uniform(distribution)
        measure.gen_mimic_samples(n_min=0,n_max=4)
    
    def test_transform_g_to_f(self):
        # implicitly called from Integrand superclass constructor
        distribution = Sobol(dimension=3)
        measure = Uniform(distribution)
        Keister(measure) 

    def test_set_dimension(self):
        # default params
        distribution = Sobol(dimension=2)
        measure = Uniform(distribution)
        measure.set_dimension(3)
        samples = measure.gen_mimic_samples(4)
        self.assertTrue(samples.shape==(4,3))
        # other compatible parameters scheme
        distribution = Sobol(dimension=2)
        measure = Uniform(distribution, lower_bound=[-2,-2],upper_bound=[2,2])
        measure.set_dimension(3)
        samples = measure.gen_mimic_samples(4)
        self.assertTrue(samples.shape==(4,3))
        self.assertTrue((measure.lower_bound==tile(-2,3)).all() and (measure.upper_bound==tile(2,3)).all())
        # bad parameters
        distribution = Sobol(dimension=2)
        measure = Uniform(distribution, lower_bound=[0,-2],upper_bound=[2,2])
        self.assertRaises(DimensionError,measure.set_dimension,3)
        # bad parameters
        distribution = Sobol(dimension=2)
        measure = Uniform(distribution, lower_bound=[-2,-2],upper_bound=[5,2])
        self.assertRaises(DimensionError,measure.set_dimension,3)
        

class TestGaussian(unittest.TestCase):
    """ Unit tests for Gaussian Measure. """

    def test_gen_mimic_samples(self):
        distribution = Sobol(dimension=3)
        measure = Gaussian(distribution)
        measure.gen_mimic_samples(n_min=0,n_max=4)

    def test_transform_g_to_f(self):
        # implicitly called from Integrand superclass constructor
        distribution = Sobol(dimension=3)
        measure = Gaussian(distribution)
        Keister(measure) 

    def test_set_dimension(self):
        # default params
        distribution = Sobol(dimension=2)
        measure = Gaussian(distribution)
        measure.set_dimension(3)
        samples = measure.gen_mimic_samples(4)
        self.assertTrue(samples.shape==(4,3))
        # other compatible parameters scheme
        distribution = Sobol(dimension=2)
        measure = Gaussian(distribution, mean=[1,1],covariance=[2,2])
        measure.set_dimension(3)
        samples = measure.gen_mimic_samples(4)
        self.assertTrue(samples.shape==(4,3))
        self.assertTrue((measure.mu==tile(1,3)).all() and (measure.sigma2==2*eye(3)).all())
        # other compatible parameters scheme
        distribution = Sobol(dimension=2)
        measure = Gaussian(distribution, mean=[1,1],covariance=2*eye(2))
        measure.set_dimension(3)
        samples = measure.gen_mimic_samples(4)
        self.assertTrue(samples.shape==(4,3))
        self.assertTrue((measure.mu==tile(1,3)).all() and (measure.sigma2==2*eye(3)).all())
        # bad parameters
        distribution = Sobol(dimension=2)
        measure = Gaussian(distribution, mean=[1,1],covariance=[1,2])
        self.assertRaises(DimensionError,measure.set_dimension,3)
        # bad parameters
        distribution = Sobol(dimension=2)
        measure = Gaussian(distribution, mean=[1,2],covariance=[2,2])
        self.assertRaises(DimensionError,measure.set_dimension,3)


class TestBrownianMontion(unittest.TestCase):
    """ Unit tests for Brownian Motion Measure. """

    def test_gen_mimic_samples(self):
        distribution = Sobol(dimension=3)
        measure = BrownianMotion(distribution)
        measure.gen_mimic_samples(n_min=0,n_max=4)

    def test_transform_g_to_f(self):
        # implicitly called from Integrand superclass constructor
        distribution = Sobol(dimension=3)
        measure = BrownianMotion(distribution)
        Keister(measure)
    
    def test_set_dimension(self):
        distribution = Sobol(dimension=2)
        measure = BrownianMotion(distribution)
        measure.set_dimension(4)
        samples = measure.gen_mimic_samples(8)
        self.assertTrue(samples.shape==(8,4))
        self.assertTrue((measure.time_vector==array([1./4,1./2,3./4,1.])).all())


class TestLebesgue(unittest.TestCase):
    """ Unit tests for Lebesgue Measure. """

    def test_transform_g_to_f(self):
        # implicitly called from Integrand superclass constructor
        distribution = Sobol(dimension=3)
        measure = Lebesgue(distribution)
        Keister(measure)
    
    def test_set_dimension(self):
        distribution = Sobol(dimension=2)
        measure = Lebesgue(distribution)
        self.assertRaises(DimensionError,measure.set_dimension,3)


class TestIdentityTransform(unittest.TestCase):
    """ Unit tests for IdentityTransform Measure. """

    def test_gen_mimic_samples(self):
        distribution = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,2)))
        measure = IdentityTransform(distribution)
        samples = measure.gen_mimic_samples(n=5)
    
    def test_transform_g_to_f(self):
        # implicitly called from Integrand superclass constructor
        distribution = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,2)))
        measure = IdentityTransform(distribution)
        Keister(measure)
    
    def test_set_dimension(self):
        distribution = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,2)))
        measure = IdentityTransform(distribution)
        self.assertRaises(DimensionError,measure.set_dimension,3)


class TestImportanceSampling(unittest.TestCase):
    """ Unit tests for ImportanceSampling Measure. """
    
    def test_construct(self):
        def quarter_circle_uniform_pdf(x):
            # see sampling measures demo
            x1,x2 = x
            if sqrt(x1**2+x2**2)<1 and x1>=0 and x2>=0:
                return 4/pi 
            else:
                return 0. # outside of quarter circle
        measure = ImportanceSampling(
            objective_pdf = quarter_circle_uniform_pdf,
            measure_to_sample_from = Uniform(Lattice(dimension=2,seed=9)))


if __name__ == "__main__":
    unittest.main()
