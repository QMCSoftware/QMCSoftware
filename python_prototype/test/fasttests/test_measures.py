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
        measure = BrownianMotion(distribution, time_vector=[1/3,2/3,1])
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

    def test_gen_samples(self):
        distribution = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,2)))
        measure = IdentityTransform(distribution)
        samples = measure.gen_mimic_samples(n=5)


class TestInverseCDFTransform(unittest.TestCase):

    def test_gen_samples(self):
        distribution = Sobol(2)
        exp_lambda_5 = InverseCDFTransform(distribution,\
            inverse_cdf_fun=lambda u,l=5: log(1-u)/(-l))
        exp_lambda_5.gen_mimic_samples(n_min=0,n_max=4)

if __name__ == "__main__":
    unittest.main()
