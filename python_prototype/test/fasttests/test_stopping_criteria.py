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
        distribution = Lattice(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CLT, integrand)
        distribution = Sobol(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CLT, integrand)

    def test_n_max_single_level(self):
        distribution = IIDStdUniform(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        algorithm = CLT(integrand, abs_tol=.001, n_init=64, n_max=1000)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
        
    def test_keister_2d(self):
        distribution = IIDStdUniform(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        solution,data = CLT(integrand, abs_tol=abs_tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < abs_tol)


class TestCltRep(unittest.TestCase):
    """
    Unit tests for CltRep in QMCPy.
    """

    def test_raise_distribution_compatibility_error(self):
        distribution = IIDStdGaussian(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CLTRep, integrand)
        distribution = IIDStdUniform(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CLTRep, integrand)

    def test_n_max_single_level(self):
        distribution = Lattice(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        algorithm = CLTRep(integrand, abs_tol=.001, n_init=16, n_max=32)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_keister_2d(self):
        distribution = Sobol(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        solution,data = CLTRep(integrand, abs_tol=abs_tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < abs_tol)


class TestMeanMC_g(unittest.TestCase):
    """
    Unit tests for MeanMC_g in QMCPy
    """

    def test_raise_distribution_compatibility_error(self):
        distribution = Lattice(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, MeanMC_g, integrand)
        distribution = Sobol(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, MeanMC_g, integrand)

    def test_n_max_single_level(self):
        distribution = IIDStdUniform(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        algorithm = MeanMC_g(integrand, abs_tol=.001, n_init=64, n_max=500)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    
    def test_keister_2d(self):
        distribution = IIDStdGaussian(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        solution,data = MeanMC_g(integrand, abs_tol=abs_tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < abs_tol)


class TestCubLattice_g(unittest.TestCase):
    """
    Unit tests for CubLattice_g in QMCPy
    """

    def test_raise_distribution_compatibility_error(self):
        distribution = IIDStdGaussian(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CubLattice_g, integrand)
        distribution = IIDStdUniform(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CubLattice_g, integrand)
        distribution = Sobol(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CubLattice_g, integrand)

    def test_n_max_single_level(self):
        distribution = Lattice(dimension=2, backend="GAIL")
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        algorithm = CubLattice_g(integrand, abs_tol=.001, n_init=2**8, n_max=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_keister_2d(self):
        distribution = Lattice(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        solution,data = CubLattice_g(integrand, abs_tol=abs_tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < abs_tol)


class TestCubSobol_g(unittest.TestCase):
    """
    Unit tests for CubSobol_g in QMCPy
    """

    def test_raise_distribution_compatibility_error(self):
        distribution = IIDStdGaussian(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CubSobol_g, integrand)
        distribution = IIDStdUniform(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CubSobol_g, integrand)
        distribution = Lattice(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CubSobol_g, integrand)

    def test_n_max_single_level(self):
        distribution = Sobol(dimension=2, backend="QRNG")
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        algorithm = CubSobol_g(integrand, abs_tol=.001, n_init=2**8, n_max=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_keister_2d(self):
        distribution = Sobol(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        solution,data = CubSobol_g(integrand, abs_tol=abs_tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < abs_tol)
        
        
if __name__ == "__main__":
    unittest.main()
