""" Unit tests for subclasses of StoppingCriterion in QMCPy """

from qmcpy import *
from qmcpy.util import *
import sys
import numpy
import unittest

keister_2d_exact = 1.808186429263620
tol = .05
rel_tol = 0


class TestCubMCCLT(unittest.TestCase):
    """ Unit tests for CubMCCLT StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(Lattice(dimension=2))
        self.assertRaises(DistributionCompatibilityError, CubMCCLT, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(IIDStdUniform(dimension=2))
        algorithm = CubMCCLT(integrand, abs_tol=.001, n_init=64, n_max=1000)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
        
    def test_keister_2d(self):
        integrand = Keister(IIDStdUniform(dimension=2))
        solution,data = CubMCCLT(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubQMCCLT(unittest.TestCase):
    """ Unit tests for CubQMCCLT StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2))
        self.assertRaises(DistributionCompatibilityError, CubQMCCLT, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Lattice(dimension=2))
        algorithm = CubQMCCLT(integrand, abs_tol=.001, n_init=16, n_max=32)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_keister_2d(self):
        integrand = Keister(Halton(dimension=2))
        solution,data = CubQMCCLT(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubMCG(unittest.TestCase):
    """ Unit tests for CubMCG StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(Lattice(dimension=2))
        self.assertRaises(DistributionCompatibilityError, CubMCG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(IIDStdUniform(dimension=2))
        algorithm = CubMCG(integrand, abs_tol=.001, n_init=64, n_max=500)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    
    def test_keister_2d(self):
        integrand = Keister(IIDStdUniform(dimension=2))
        solution,data = CubMCG(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubQMCLatticeG(unittest.TestCase):
    """ Unit tests for CubQMCLatticeG StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2))
        self.assertRaises(DistributionCompatibilityError, CubQMCLatticeG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Lattice(dimension=2))
        algorithm = CubQMCLatticeG(integrand, abs_tol=.001, n_init=2**8, n_max=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_keister_2d(self):
        integrand = Keister(Lattice(dimension=2))
        solution,data = CubQMCLatticeG(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubQMCSobolG(unittest.TestCase):
    """ Unit tests for CubQMCSobolG StoppingCriterion. """
    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2))
        self.assertRaises(DistributionCompatibilityError, CubQMCSobolG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Sobol(dimension=2))
        algorithm = CubQMCSobolG(integrand, abs_tol=.001, n_init=2**8, n_max=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_keister_2d(self):
        integrand = Keister(Sobol(dimension=2))
        solution,data = CubQMCSobolG(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubMCL(unittest.TestCase):
    """ Unit tests for CubMCML StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = MLCallOptions(Lattice())
        self.assertRaises(DistributionCompatibilityError, CubMCML, integrand)

    def test_n_max(self):
        integrand = MLCallOptions(IIDStdUniform(),start_strike_price=30)
        algorithm = CubMCML(integrand,rmse_tol=.001,n_max=2**10)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_european_option(self):
        integrand = MLCallOptions(IIDStdUniform(),start_strike_price=30)
        solution,data = CubMCML(integrand,rmse_tol=tol/2.58).integrate()
        exact_value = integrand.get_exact_value()
        self.assertTrue(abs(solution-exact_value) < tol)


class TestCubQMCML(unittest.TestCase):
    """ Unit tests for CubQMCML StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = MLCallOptions(IIDStdUniform())
        self.assertRaises(DistributionCompatibilityError, CubQMCML, integrand)

    def test_n_max(self):
        integrand = MLCallOptions(Lattice(),start_strike_price=30)
        algorithm = CubQMCML(integrand,rmse_tol=tol/2.58,n_max=2**10)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)
    
    def test_european_option(self):
        integrand = MLCallOptions(Halton(),start_strike_price=30)
        solution,data = CubQMCML(integrand,rmse_tol=tol/2.58).integrate()
        exact_value = integrand.get_exact_value()
        self.assertTrue(abs(solution-exact_value) < tol)


class TestCubBayesLatticeG(unittest.TestCase):
    """ Unit tests for CubBayesLatticeG StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2))
        self.assertRaises(DistributionCompatibilityError, CubBayesLatticeG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Lattice(dimension=2, order='linear'))
        algorithm = CubBayesLatticeG(integrand, abs_tol=.0001, n_init=2 ** 8, n_max=2 ** 9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(Lattice(dimension=2, order='linear'))
        solution, data = CubBayesLatticeG(integrand, abs_tol=tol, n_init=2 ** 5).integrate()
        self.assertTrue(abs(solution - keister_2d_exact) < tol)


class TestCubBayesNetG(unittest.TestCase):
    """ Unit tests for CubBayesNetG StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2))
        self.assertRaises(DistributionCompatibilityError, CubBayesNetG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Sobol(dimension=2))
        algorithm = CubBayesNetG(integrand, abs_tol=.0001, n_init=2 ** 8, n_max=2 ** 9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(Sobol(dimension=2, randomize='LMS', graycode=False))
        solution, data = CubBayesNetG(integrand , n_init=2 ** 5, abs_tol=tol).integrate()  #
        self.assertTrue(abs(solution - keister_2d_exact) < tol)


if __name__ == "__main__":
    unittest.main()
