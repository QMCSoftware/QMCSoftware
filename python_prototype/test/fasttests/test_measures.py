""" Unit tests for subclasses of TrueMeasures in QMCPy """

from qmcpy import *
from numpy import *
import unittest

        
class TestUniform(unittest.TestCase):
    """
    Unit tests for Uniform distribution in QMCPy.
    """

    def test_gen_samples(self):
        distribution = Sobol(dimension=3)
        measure = Uniform(distribution)
        measure.gen_mimic_samples(n_min=0,n_max=4)


class TestGaussian(unittest.TestCase):
    """
    Unit tests for Gaussian distribution in QMCPy.
    """

    def test_gen_samples(self):
        distribution = Sobol(dimension=3)
        measure = Gaussian(distribution)
        measure.gen_mimic_samples(n_min=0,n_max=4)


class TestBrownianMontion(unittest.TestCase):
    """
    Unit tests for Brownian Motion in QMCPy.
    """

    def test_gen_samples(self):
        distribution = Sobol(dimension=3)
        measure = BrownianMotion(distribution)
        measure.gen_mimic_samples(n_min=0,n_max=4)


class TestLebesgue(unittest.TestCase):
    """
    Unit tests for Lebesgue in QMCPy.
    """

    def test_gen_samples(self):
        distribution = Sobol(dimension=3)
        measure = Lebesgue(distribution)
        self.assertRaises(Exception,measure.gen_mimic_samples,n_min=0,n_max=4)

class TestIdentityTransform(unittest.TestCase):
    """
    Unit tests for IdentityTransform
    """

    def test_gen_samples(self):
        distribution = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,2)))
        measure = IdentityTransform(distribution)
        samples = measure.gen_mimic_samples(n=5)


class TestImportanceSampling(unittest.TestCase):
    """
    Unit tests for ImportanceSampling
    """
    
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
