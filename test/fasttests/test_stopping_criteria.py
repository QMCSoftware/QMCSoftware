""" Unit tests for subclasses of StoppingCriterion in QMCPy """

from qmcpy import *
from qmcpy.util import *
from numpy import arange
import unittest

keister_2d_exact = 1.808186429263620
tol = .05
rel_tol = 0


class TestCubMcClt(unittest.TestCase):
    """ Unit tests for CubMcClt StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        distribution = Lattice(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CubMcClt, integrand)

    def test_n_max_single_level(self):
        distribution = IIDStdUniform(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        algorithm = CubMcClt(integrand, abs_tol=.001, n_init=64, n_max=1000)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
        
    def test_keister_2d(self):
        distribution = IIDStdUniform(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        solution,data = CubMcClt(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubQmcClt(unittest.TestCase):
    """ Unit tests for CubQmcClt StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        distribution = IIDStdGaussian(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CubQmcClt, integrand)

    def test_n_max_single_level(self):
        distribution = Lattice(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        algorithm = CubQmcClt(integrand, abs_tol=.001, n_init=16, n_max=32)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_keister_2d(self):
        distribution = Sobol(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        solution,data = CubQmcClt(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubMcG(unittest.TestCase):
    """ Unit tests for CubMcG StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        distribution = Lattice(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CubMcG, integrand)

    def test_n_max_single_level(self):
        distribution = IIDStdUniform(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        algorithm = CubMcG(integrand, abs_tol=.001, n_init=64, n_max=500)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    
    def test_keister_2d(self):
        distribution = IIDStdGaussian(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        solution,data = CubMcG(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubQmcLatticeG(unittest.TestCase):
    """ Unit tests for CubQmcLatticeG StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        distribution = IIDStdGaussian(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CubQmcLatticeG, integrand)

    def test_n_max_single_level(self):
        distribution = Lattice(dimension=2, backend="GAIL")
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        algorithm = CubQmcLatticeG(integrand, abs_tol=.001, n_init=2**8, n_max=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_keister_2d(self):
        distribution = Lattice(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        solution,data = CubQmcLatticeG(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubQmcSobolG(unittest.TestCase):
    """ Unit tests for CubQmcSobolG StoppingCriterion. """
    def test_raise_distribution_compatibility_error(self):
        distribution = IIDStdGaussian(dimension=2)
        measure = Gaussian(distribution)
        integrand = Keister(measure)
        self.assertRaises(DistributionCompatibilityError, CubQmcSobolG, integrand)

    def test_n_max_single_level(self):
        distribution = Sobol(dimension=2, backend="QRNG")
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        algorithm = CubQmcSobolG(integrand, abs_tol=.001, n_init=2**8, n_max=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_keister_2d(self):
        distribution = Sobol(dimension=2)
        measure = Gaussian(distribution, covariance=1/2)
        integrand = Keister(measure)
        solution,data = CubQmcSobolG(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubMcMl(unittest.TestCase):
    """ Unit tests for CubMcMl StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = MLCallOptions(Gaussian(Lattice()))
        self.assertRaises(DistributionCompatibilityError, CubMcMl, integrand)

    def test_n_max(self):
        integrand = MLCallOptions(Gaussian(IIDStdUniform()),start_strike_price=30)
        algorithm = CubMcMl(integrand,rmse_tol=.001,n_max=2**10)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_european_option(self):
        integrand = MLCallOptions(Gaussian(IIDStdUniform()),start_strike_price=30)
        solution,data = CubMcMl(integrand,rmse_tol=tol/2.58).integrate()
        exact_value = integrand.get_exact_value()
        self.assertTrue(abs(solution-exact_value) < tol)


class TestCubQmcMl(unittest.TestCase):
    """ Unit tests for CubQmcMl StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = MLCallOptions(Gaussian(IIDStdGaussian()))
        self.assertRaises(DistributionCompatibilityError, CubQmcMl, integrand)

    def test_n_max(self):
        integrand = MLCallOptions(Gaussian(Lattice()),start_strike_price=30)
        algorithm = CubQmcMl(integrand,rmse_tol=tol/2.58,n_max=2**10)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_european_option(self):
        integrand = MLCallOptions(Gaussian(Sobol()),start_strike_price=30)
        solution,data = CubQmcMl(integrand,rmse_tol=tol/2.58).integrate()
        exact_value = integrand.get_exact_value()
        self.assertTrue(abs(solution-exact_value) < tol)


if __name__ == "__main__":
    unittest.main()
