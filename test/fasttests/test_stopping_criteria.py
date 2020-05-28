""" Unit tests for subclasses of StoppingCriterion in QMCPy """

from qmcpy import *
from qmcpy.util import *
from numpy import arange
import unittest

keister_2d_exact = 1.808186429263620
tol = .05
rel_tol = 0


class TestClt(unittest.TestCase):
    """ Unit tests for Clt StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        distribution = Lattice(dimension=2)
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
        solution,data = CLT(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCltRep(unittest.TestCase):
    """ Unit tests for CltRep StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        distribution = IIDStdGaussian(dimension=2)
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
        solution,data = CLTRep(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestMeanMC_g(unittest.TestCase):
    """ Unit tests for MeanMC_g StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        distribution = Lattice(dimension=2)
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
        solution,data = MeanMC_g(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubLattice_g(unittest.TestCase):
    """ Unit tests for CubLattice_g StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        distribution = IIDStdGaussian(dimension=2)
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
        solution,data = CubLattice_g(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubSobol_g(unittest.TestCase):
    """ Unit tests for CubSobol_g StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        distribution = IIDStdGaussian(dimension=2)
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
        solution,data = CubSobol_g(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestMLMC(unittest.TestCase):
    """ Unit tests for MLMC StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = MLMCCallOptions(Gaussian(Lattice()))
        self.assertRaises(DistributionCompatibilityError, MLMC, integrand)

    def test_n_max(self):
        integrand = MLMCCallOptions(Gaussian(IIDStdUniform()),start_strike_price=30)
        algorithm = MLMC(integrand,rmse_tol=.001,n_max=2**10)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_european_option(self):
        integrand = MLMCCallOptions(Gaussian(IIDStdUniform()),start_strike_price=30)
        solution,data = MLMC(integrand,rmse_tol=tol/2.58).integrate()
        exact_value = integrand.get_exact_value()
        self.assertTrue(abs(solution-exact_value) < tol)


class TestMLQMC(unittest.TestCase):
    """ Unit tests for MLQMC StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = MLMCCallOptions(Gaussian(IIDStdGaussian()))
        self.assertRaises(DistributionCompatibilityError, MLQMC, integrand)

    def test_n_max(self):
        integrand = MLMCCallOptions(Gaussian(Lattice()),start_strike_price=30)
        algorithm = MLQMC(integrand,rmse_tol=tol/2.58,n_max=2**10)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_european_option(self):
        integrand = MLMCCallOptions(Gaussian(Sobol()),start_strike_price=30)
        solution,data = MLQMC(integrand,rmse_tol=tol/2.58).integrate()
        exact_value = integrand.get_exact_value()
        self.assertTrue(abs(solution-exact_value) < tol)


if __name__ == "__main__":
    unittest.main()
