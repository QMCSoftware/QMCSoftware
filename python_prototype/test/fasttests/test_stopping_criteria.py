""" Unit tests for subclasses of StoppingCriterion in QMCPy """

from qmcpy import *
from qmcpy.util import *
from numpy import arange
import unittest

keister_2d_exact = 1.808186429263620
abs_tol = .01
rel_tol = 0


class TestClt(unittest.TestCase):
    """
    Unit tests for Clt in QMCPy.
    """

    def test_raise_distribution_compatibility_error(self):
        self.assertRaises(DistributionCompatibilityError, CLT, Lattice(dimension=2))
        self.assertRaises(DistributionCompatibilityError, CLT, Sobol(dimension=2))

    def test_n_max_single_level(self):
        distribution = IIDStdUniform(dimension=2)
        measure = Gaussian(distribution, variance=1/2)
        integrand = Keister(measure)
        algorithm = CLT(distribution, abs_tol=.001, n_init=64, n_max=1000)
        self.assertWarns(MaxSamplesWarning, integrate, \
            algorithm, integrand, measure, distribution)
        
    def test_keister_2d(self):
        distribution = IIDStdUniform(dimension=2)
        measure = Gaussian(distribution, variance=1/2)
        integrand = Keister(measure)
        algorithm = CLT(distribution, abs_tol=abs_tol)
        solution,data = integrate(algorithm, integrand, measure, distribution)
        self.assertTrue(abs(solution-keister_2d_exact) < abs_tol)


class TestCltRep(unittest.TestCase):
    """
    Unit tests for CltRep in QMCPy.
    """

    def test_raise_distribution_compatibility_error(self):
        self.assertRaises(DistributionCompatibilityError, CLTRep, IIDStdUniform(dimension=2))
        self.assertRaises(DistributionCompatibilityError, CLTRep, IIDStdGaussian(dimension=2))

    def test_n_max_single_level(self):
        distribution = Lattice(dimension=2, replications=16)
        measure = Gaussian(distribution, variance=1/2)
        integrand = Keister(measure)
        algorithm = CLTRep(distribution, abs_tol=.001, n_init=16, n_max=32)
        self.assertWarns(MaxSamplesWarning, integrate, \
            algorithm, integrand, measure, distribution)
    
    def test_keister_2d(self):
        distribution = Sobol(dimension=2, replications=16)
        measure = Gaussian(distribution, variance=1/2)
        integrand = Keister(measure)
        algorithm = CLTRep(distribution, abs_tol=abs_tol)
        solution,data = integrate(algorithm, integrand, measure, distribution)
        self.assertTrue(abs(solution-keister_2d_exact) < abs_tol)


class TestMeanMC_g(unittest.TestCase):
    """
    Unit tests for MeanMC_g in QMCPy
    """

    def test_raise_distribution_compatibility_error(self):
        self.assertRaises(DistributionCompatibilityError, MeanMC_g, Lattice(dimension=2))
        self.assertRaises(DistributionCompatibilityError, MeanMC_g, Sobol(dimension=2))

    def test_n_max_single_level(self):
        distribution = IIDStdUniform(dimension=2)
        measure = Gaussian(distribution, variance=1/2)
        integrand = Keister(measure)
        algorithm = MeanMC_g(distribution, abs_tol=.001, n_init=64, n_max=500)
        self.assertWarns(MaxSamplesWarning, integrate, \
            algorithm, integrand, measure, distribution)
    
    def test_keister_2d(self):
        distribution = IIDStdGaussian(dimension=2)
        measure = Gaussian(distribution, variance=1/2)
        integrand = Keister(measure)
        algorithm = MeanMC_g(distribution, abs_tol=abs_tol)
        solution,data = integrate(algorithm, integrand, measure, distribution)
        self.assertTrue(abs(solution-keister_2d_exact) < abs_tol)


class TestCubLattice_g(unittest.TestCase):
    """
    Unit tests for CubLattice_g in QMCPy
    """

    def test_raise_distribution_compatibility_error(self):
        self.assertRaises(DistributionCompatibilityError, CubLattice_g, IIDStdUniform(dimension=2))
        self.assertRaises(DistributionCompatibilityError, CubLattice_g, IIDStdGaussian(dimension=2))
        self.assertRaises(DistributionCompatibilityError, CubLattice_g, Sobol(dimension=2))

    def test_n_max_single_level(self):
        distribution = Lattice(dimension=2, replications=0, backend="GAIL")
        measure = Gaussian(distribution, variance=1/2)
        integrand = Keister(measure)
        algorithm = CubLattice_g(distribution, abs_tol=.001, n_init=2**8, n_max=2**9)
        self.assertWarns(MaxSamplesWarning, integrate, \
            algorithm, integrand, measure, distribution)
    
    def test_keister_2d(self):
        distribution = Lattice(dimension=2)
        measure = Gaussian(distribution, variance=1/2)
        integrand = Keister(measure)
        algorithm = CubLattice_g(distribution, abs_tol=abs_tol)
        solution,data = integrate(algorithm, integrand, measure, distribution)
        self.assertTrue(abs(solution-keister_2d_exact) < abs_tol)


if __name__ == "__main__":
    unittest.main()
