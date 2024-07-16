""" Unit tests for subclasses of StoppingCriterion in QMCPy """

from qmcpy import *
from qmcpy.util import *
from numpy import *
import unittest

keister_2d_exact = 1.808186429263620
tol = .005
rel_tol = 0


class TestCubMCCLT(unittest.TestCase):
    """ Unit tests for CubMCCLT StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(Lattice(dimension=2,seed=7))
        self.assertRaises(DistributionCompatibilityError, CubMCCLT, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(IIDStdUniform(dimension=2,seed=7))
        algorithm = CubMCCLT(integrand, abs_tol=.001, n_init=64, n_max=1000)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(IIDStdUniform(dimension=2,seed=7))
        solution,data = CubMCCLT(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubMCG(unittest.TestCase):
    """ Unit tests for CubMCG StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(Lattice(dimension=2,seed=7))
        self.assertRaises(DistributionCompatibilityError, CubMCG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(IIDStdUniform(dimension=2,seed=7))
        algorithm = CubMCG(integrand, abs_tol=.001, n_init=64, n_max=500)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(IIDStdUniform(dimension=2,seed=7))
        solution,data = CubMCG(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)


class TestCubQMCCLT(unittest.TestCase):
    """ Unit tests for CubQMCCLT StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2,seed=7))
        self.assertRaises(DistributionCompatibilityError, CubQMCCLT, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Lattice(dimension=2,seed=7))
        algorithm = CubQMCCLT(integrand, abs_tol=.001, n_init=16, n_max=32)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(Halton(dimension=2,seed=7))
        solution,data = CubQMCCLT(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact)<tol)

    def test_sobol_indices_dnb2(self):
        abs_tol,rel_tol = 5e-2,0
        a,b = 7,0.1
        indices = [[0],[1],[2],[0,1],[0,2],[1,2]]
        true_solution = Ishigami._exact_sensitivity_indices(indices,a,b)
        for i in range(3):
            si_ishigami = SensitivityIndices(Ishigami(DigitalNetB2(3,seed=7),a,b),indices)
            solution,data = CubQMCCLT(si_ishigami,abs_tol=abs_tol,rel_tol=rel_tol).integrate()
            abs_error = abs(solution.squeeze()-true_solution)
            success = (abs_error<abs_tol).all()
            if success: break
        self.assertTrue(success)

    def test_sobol_indices_lattice(self):
        abs_tol,rel_tol = 5e-2,0
        a,b = 7,0.1
        indices = [[0],[1],[2],[0,1],[0,2],[1,2]]
        true_solution = Ishigami._exact_sensitivity_indices(indices,a,b)
        for i in range(3):
            si_ishigami = SensitivityIndices(Ishigami(Lattice(3,seed=7),a,b),indices)
            solution,data = CubQMCCLT(si_ishigami,abs_tol=abs_tol,rel_tol=rel_tol).integrate()
            abs_error = abs(solution.squeeze()-true_solution)
            success = (abs_error<abs_tol).all()
            if success: break
        self.assertTrue(success)

    def test_sobol_indices_halton(self):
        abs_tol,rel_tol = 5e-2,0
        a,b = 7,0.1
        indices = [[0],[1],[2],[0,1],[0,2],[1,2]]
        true_solution = Ishigami._exact_sensitivity_indices(indices,a,b)
        for i in range(3):
            si_ishigami = SensitivityIndices(Ishigami(Halton(3,seed=7),a,b),indices)
            solution,data = CubQMCCLT(si_ishigami,abs_tol=abs_tol,rel_tol=rel_tol).integrate()
            abs_error = abs(solution.squeeze()-true_solution)
            success = (abs_error<abs_tol).all()
            if success: break
        self.assertTrue(success)


class TestCubQMCLatticeG(unittest.TestCase):
    """ Unit tests for CubQMCLatticeG StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2,seed=7))
        self.assertRaises(DistributionCompatibilityError, CubQMCLatticeG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Lattice(dimension=2,seed=7))
        algorithm = CubQMCLatticeG(integrand, abs_tol=.001, n_init=2**8, n_max=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(Lattice(dimension=2,seed=7))
        solution,data = CubQMCLatticeG(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)

    def test_sobol_indices(self):
        abs_tol,rel_tol = 1e-2,0
        a,b = 7,0.1
        indices = [[0],[1],[2],[0,1],[0,2],[1,2]]
        true_solution = Ishigami._exact_sensitivity_indices(indices,a,b)
        for i in range(3):
            si_ishigami = SensitivityIndices(Ishigami(Lattice(3,seed=7),a,b),indices)
            solution,data = CubQMCLatticeG(si_ishigami,abs_tol=abs_tol,rel_tol=rel_tol).integrate()
            abs_error = abs(solution.squeeze()-true_solution)
            success = (abs_error<abs_tol).all()
            if success: break
        self.assertTrue(success)


class TestCubQMCNetG(unittest.TestCase):
    """ Unit tests for CubQMCNetG StoppingCriterion. """
    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2,seed=7))
        self.assertRaises(DistributionCompatibilityError, CubQMCNetG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(DigitalNetB2(dimension=2,seed=7))
        algorithm = CubQMCNetG(integrand, abs_tol=.001, n_init=2**8, n_max=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(DigitalNetB2(dimension=2,seed=7))
        solution,data = CubQMCNetG(integrand, abs_tol=tol).integrate()
        self.assertTrue(abs(solution-keister_2d_exact) < tol)

    def test_sobol_indices(self):
        abs_tol,rel_tol = 1e-2,0
        a,b = 7,0.1
        indices = [[0],[1],[2],[0,1],[0,2],[1,2]]
        true_solution = Ishigami._exact_sensitivity_indices(indices,a,b)
        for i in range(3):
            si_ishigami = SensitivityIndices(Ishigami(DigitalNetB2(3,seed=7),a,b),indices)
            solution,data = CubQMCNetG(si_ishigami,abs_tol=abs_tol,rel_tol=rel_tol).integrate()
            abs_error = abs(solution.squeeze()-true_solution)
            success = (abs_error<abs_tol).all()
            if success: break
        self.assertTrue(success)

class TestCubBayesLatticeG(unittest.TestCase):
    """ Unit tests for CubBayesLatticeG StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2,seed=7))
        self.assertRaises(DistributionCompatibilityError, CubBayesLatticeG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Lattice(dimension=2, seed=7, order='linear'))
        algorithm = CubBayesLatticeG(integrand, abs_tol=.0001, n_init=2 ** 8, n_max=2 ** 9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(Lattice(dimension=2,seed=7, order='linear'))
        solution, data = CubBayesLatticeG(integrand, abs_tol=tol, n_init=2 ** 5).integrate()
        self.assertTrue(abs(solution - keister_2d_exact) < tol)

    def test_sobol_indices_bayes_lattice(self, dims=3, abs_tol=1e-2):
        keister_d_ = Keister(Lattice(dimension=dims, seed=7))
        keister_indices_ = SobolIndices(keister_d_, indices='singletons')
        sc_ = CubQMCLatticeG(keister_indices_, abs_tol=abs_tol, ptransform='Baker')
        solution_, data_ = sc_.integrate()

        keister_d = Keister(Lattice(dimension=dims, order='linear', seed=7))
        keister_indices = SobolIndices(keister_d, indices='singletons')
        sc = CubBayesLatticeG(keister_indices, order=1, abs_tol=abs_tol, ptransform='Baker')
        solution, data = sc.integrate()

        self.assertTrue(solution.shape, (dims, dims, 1))
        self.assertTrue(abs(solution - solution_).max() < abs_tol)

    def test_sobol_indices(self):
        abs_tol,rel_tol = 5e-2,0
        a,b = 7,0.1
        indices = [[0],[1],[2],[0,1],[0,2],[1,2]]
        true_solution = Ishigami._exact_sensitivity_indices(indices,a,b)
        for i in range(3):
            si_ishigami = SensitivityIndices(Ishigami(Lattice(3,seed=7,order='linear'),a,b),indices)
            solution,data = CubBayesLatticeG(si_ishigami,abs_tol=abs_tol,rel_tol=rel_tol,order=1,ptransform='Baker').integrate()
            abs_error = abs(solution.squeeze()-true_solution)
            success = (abs_error<abs_tol).all()
            if success: break
        self.assertTrue(success)


class TestCubBayesNetG(unittest.TestCase):
    """ Unit tests for CubBayesNetG StoppingCriterion. """

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2,seed=7))
        self.assertRaises(DistributionCompatibilityError, CubBayesNetG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(DigitalNetB2(dimension=2,seed=7))
        algorithm = CubBayesNetG(integrand, abs_tol=.0001, n_init=2 ** 8, n_max=2 ** 9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(DigitalNetB2(dimension=2,seed=7))
        solution, data = CubBayesNetG(integrand , n_init=2 ** 5, abs_tol=tol).integrate()
        self.assertTrue(abs(solution - keister_2d_exact) < tol)

    def test_sobol_indices(self):
        abs_tol,rel_tol = 5e-2,0
        a,b = 7,0.1
        indices = [[0],[1],[2],[0,1],[0,2],[1,2]]
        true_solution = Ishigami._exact_sensitivity_indices(indices,a,b)
        for i in range(3):
            si_ishigami = SensitivityIndices(Ishigami(DigitalNetB2(3,seed=7),a,b),indices)
            solution,data = CubBayesNetG(si_ishigami,abs_tol=abs_tol,rel_tol=rel_tol).integrate()
            abs_error = abs(solution.squeeze()-true_solution)
            success = (abs_error<abs_tol).all()
            if success: break
        self.assertTrue(success)


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
        algorithm = CubQMCML(integrand,abs_tol=tol,n_max=2**10)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_european_option(self):
        integrand = MLCallOptions(Halton(),start_strike_price=30)
        solution,data = CubQMCML(integrand,abs_tol=tol).integrate()
        exact_value = integrand.get_exact_value()
        self.assertTrue(abs(solution-exact_value) < tol)


if __name__ == "__main__":
    unittest.main()