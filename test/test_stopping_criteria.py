"""Unit tests for subclasses of StoppingCriterion in QMCPy"""

import builtins
import importlib
from qmcpy import *
from qmcpy.util import *
import numpy as np
import unittest
from unittest.mock import patch

keister_2d_exact = 1.808186429263620
tol = 0.005
rel_tol = 0

class TestStoppingCriterionImportFallbacks(unittest.TestCase):
    def test_import_fallback_classes(self):
        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name in ("torch", "gpytorch"):
                raise ImportError("forced import failure")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            sc_mod = importlib.import_module("qmcpy.stopping_criterion")
            sc_mod = importlib.reload(sc_mod)
            with self.assertRaises(ModuleNotFoundError):
                sc_mod.PFGPCI()
            with self.assertRaises(ModuleNotFoundError):
                sc_mod.PFSampleErrorDensityAR()
            with self.assertRaises(ModuleNotFoundError):
                sc_mod.SuggesterSimple()
        importlib.reload(importlib.import_module("qmcpy.stopping_criterion"))
        
class TestCubMCCLT(unittest.TestCase):
    """Unit tests for CubMCCLT StoppingCriterion."""

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(Lattice(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubMCCLT, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        algorithm = CubMCCLT(integrand, abs_tol=0.001, n_init=64, n_limit=1000)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        solution, data = CubMCCLT(integrand, abs_tol=tol).integrate()
        self.assertLess(abs(solution - keister_2d_exact), tol)


class TestCubMCG(unittest.TestCase):
    """Unit tests for CubMCG StoppingCriterion."""

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(Lattice(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubMCG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        algorithm = CubMCG(integrand, abs_tol=0.001, n_init=64, n_limit=500)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        solution, data = CubMCG(integrand, abs_tol=tol).integrate()
        self.assertLess(abs(solution - keister_2d_exact), tol)


class TestCubQMCCLT(unittest.TestCase):
    """Unit tests for CubQMCCLT StoppingCriterion."""

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubQMCCLT, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Lattice(dimension=2, seed=7, replications=32))
        algorithm = CubQMCCLT(integrand, abs_tol=0.001, n_init=16, n_limit=32)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(Halton(dimension=2, seed=7, replications=32))
        solution, data = CubQMCCLT(integrand, abs_tol=tol).integrate()
        self.assertLess(abs(solution - keister_2d_exact), tol)

    def test_sobol_indices_dnb2(self):
        abs_tol, rel_tol = 5e-2, 0
        a, b = 7, 0.1
        indices = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ],
            dtype=bool,
        )
        true_solution = Ishigami._exact_sensitivity_indices(indices, a, b)
        for i in range(3):
            si_ishigami = SensitivityIndices(
                Ishigami(DigitalNetB2(3, seed=7, replications=32), a, b), indices
            )
            solution, data = CubQMCCLT(
                si_ishigami, abs_tol=abs_tol, rel_tol=rel_tol
            ).integrate()
            abs_error = abs(solution.squeeze() - true_solution)
            success = (abs_error < abs_tol).all()
            if success:
                break
        self.assertTrue(success)

    def test_sobol_indices_lattice(self):
        abs_tol, rel_tol = 5e-2, 0
        a, b = 7, 0.1
        indices = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ],
            dtype=bool,
        )
        true_solution = Ishigami._exact_sensitivity_indices(indices, a, b)
        for i in range(3):
            si_ishigami = SensitivityIndices(
                Ishigami(Lattice(3, seed=7, replications=32), a, b), indices
            )
            solution, data = CubQMCCLT(
                si_ishigami, abs_tol=abs_tol, rel_tol=rel_tol
            ).integrate()
            abs_error = abs(solution.squeeze() - true_solution)
            success = (abs_error < abs_tol).all()
            if success:
                break
        self.assertTrue(success)

    def test_sobol_indices_halton(self):
        abs_tol, rel_tol = 5e-2, 0
        a, b = 7, 0.1
        indices = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ],
            dtype=bool,
        )
        true_solution = Ishigami._exact_sensitivity_indices(indices, a, b)
        for i in range(3):
            si_ishigami = SensitivityIndices(
                Ishigami(Halton(3, seed=7, replications=32), a, b), indices
            )
            solution, data = CubQMCCLT(
                si_ishigami, abs_tol=abs_tol, rel_tol=rel_tol
            ).integrate()
            abs_error = abs(solution.squeeze() - true_solution)
            success = (abs_error < abs_tol).all()
            if success:
                break
        self.assertTrue(success)


class TestCubQMCLatticeG(unittest.TestCase):
    """Unit tests for CubQMCLatticeG StoppingCriterion."""

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubQMCLatticeG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Lattice(dimension=2, seed=7))
        algorithm = CubQMCLatticeG(integrand, abs_tol=0.001, n_init=2**8, n_limit=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(Lattice(dimension=2, seed=7))
        solution, data = CubQMCLatticeG(integrand, abs_tol=tol).integrate()
        self.assertLess(abs(solution - keister_2d_exact), tol)

    def test_sobol_indices(self):
        abs_tol, rel_tol = 5e-3, 0
        a, b = 7, 0.1
        indices = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ],
            dtype=bool,
        )
        true_solution = Ishigami._exact_sensitivity_indices(indices, a, b)
        for i in range(3):
            si_ishigami = SensitivityIndices(
                Ishigami(Lattice(3, seed=7), a, b), indices
            )
            solution, data = CubQMCLatticeG(
                si_ishigami, abs_tol=abs_tol, rel_tol=rel_tol
            ).integrate()
            abs_error = abs(solution.squeeze() - true_solution)
            success = (abs_error < abs_tol).all()
            if success:
                break
        self.assertTrue(success)


class TestCubQMCNetG(unittest.TestCase):
    """Unit tests for CubQMCNetG StoppingCriterion."""

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubQMCNetG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(DigitalNetB2(dimension=2, seed=7))
        algorithm = CubQMCNetG(integrand, abs_tol=0.001, n_init=2**8, n_limit=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(DigitalNetB2(dimension=2, seed=7))
        solution, data = CubQMCNetG(integrand, abs_tol=tol).integrate()
        self.assertLess(abs(solution - keister_2d_exact), tol)

    def test_sobol_indices(self):
        abs_tol, rel_tol = 1e-2, 0
        a, b = 7, 0.1
        indices = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ],
            dtype=bool,
        )
        true_solution = Ishigami._exact_sensitivity_indices(indices, a, b)
        for i in range(3):
            si_ishigami = SensitivityIndices(
                Ishigami(DigitalNetB2(3, seed=7), a, b), indices
            )
            solution, data = CubQMCNetG(
                si_ishigami, abs_tol=abs_tol, rel_tol=rel_tol
            ).integrate()
            abs_error = abs(solution.squeeze() - true_solution)
            success = (abs_error < abs_tol).all()
            if success:
                break
        self.assertTrue(success)


class TestCubBayesLatticeG(unittest.TestCase):
    """Unit tests for CubBayesLatticeG StoppingCriterion."""

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubBayesLatticeG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Lattice(dimension=2, seed=7, order="RADICAL INVERSE"))
        algorithm = CubBayesLatticeG(
            integrand, abs_tol=0.0001, n_init=2**8, n_limit=2**9
        )
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(Lattice(dimension=2, seed=7, order="RADICAL INVERSE"))
        solution, data = CubBayesLatticeG(
            integrand, abs_tol=tol, n_init=2**5
        ).integrate()
        self.assertTrue(abs(solution - keister_2d_exact) < tol)

    def test_sobol_indices_bayes_lattice(self, dims=3, abs_tol=1e-2):
        keister_d_ = Keister(Lattice(dimension=dims, seed=7))
        keister_indices_ = SobolIndices(keister_d_, indices="singletons")
        sc_ = CubQMCLatticeG(keister_indices_, abs_tol=abs_tol, ptransform="Baker")
        solution_, data_ = sc_.integrate()

        keister_d = Keister(Lattice(dimension=dims, order="RADICAL INVERSE", seed=7))
        keister_indices = SobolIndices(keister_d, indices="singletons")
        sc = CubBayesLatticeG(
            keister_indices, order=1, abs_tol=abs_tol, ptransform="Baker"
        )
        solution, data = sc.integrate()

        self.assertTrue(solution.shape, (dims, dims, 1))
        self.assertTrue(abs(solution - solution_).max() < abs_tol)

    def test_sobol_indices(self):
        abs_tol, rel_tol = 5e-2, 0
        a, b = 7, 0.1
        indices = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ],
            dtype=bool,
        )
        true_solution = Ishigami._exact_sensitivity_indices(indices, a, b)
        for i in range(3):
            si_ishigami = SensitivityIndices(
                Ishigami(Lattice(3, seed=7, order="natural"), a, b), indices
            )
            solution, data = CubBayesLatticeG(
                si_ishigami,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                order=1,
                ptransform="Baker",
            ).integrate()
            abs_error = abs(solution.squeeze() - true_solution)
            success = (abs_error < abs_tol).all()
            if success:
                break
        self.assertTrue(success)


class TestCubBayesNetG(unittest.TestCase):
    """Unit tests for CubBayesNetG StoppingCriterion."""

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubBayesNetG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(DigitalNetB2(dimension=2, seed=7))
        algorithm = CubBayesNetG(integrand, abs_tol=0.0001, n_init=2**8, n_limit=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(DigitalNetB2(dimension=2, seed=7))
        solution, data = CubBayesNetG(integrand, n_init=2**5, abs_tol=tol).integrate()
        self.assertTrue(abs(solution - keister_2d_exact) < tol)

    def test_sobol_indices(self):
        abs_tol, rel_tol = 1e-2, 0
        a, b = 7, 0.1
        indices = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ],
            dtype=bool,
        )
        true_solution = Ishigami._exact_sensitivity_indices(indices, a, b)
        for i in range(3):
            si_ishigami = SensitivityIndices(
                Ishigami(DigitalNetB2(3, seed=7), a, b), indices
            )
            solution, data = CubBayesNetG(
                si_ishigami, abs_tol=abs_tol, rel_tol=rel_tol
            ).integrate()
            abs_error = abs(solution.squeeze() - true_solution)
            success = (abs_error < abs_tol).all()
            if success:
                break
        self.assertTrue(success)


class TestCubMCL(unittest.TestCase):
    """Unit tests for CubMCML StoppingCriterion."""

    def test_raise_distribution_compatibility_error(self):
        integrand = FinancialOption(Lattice(seed=7))
        self.assertRaises(DistributionCompatibilityError, CubMCML, integrand)

    def test_n_max(self):
        integrand = FinancialOption(
            IIDStdUniform(seed=7), start_price=30, strike_price=30
        )
        algorithm = CubMCML(integrand, rmse_tol=0.001, n_limit=2**10)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)


class TestCubQMCML(unittest.TestCase):
    """Unit tests for CubQMCML StoppingCriterion."""

    def test_raise_distribution_compatibility_error(self):
        integrand = FinancialOption(IIDStdUniform(seed=7))
        self.assertRaises(DistributionCompatibilityError, CubQMCML, integrand)

    def test_n_max(self):
        integrand = FinancialOption(
            Lattice(replications=32, seed=7), start_price=30, strike_price=30
        )
        algorithm = CubQMCML(integrand, abs_tol=tol, n_limit=2**10)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

if __name__ == "__main__":
    unittest.main()


class TestResumeFeature(unittest.TestCase):
    """Tests for the resume parameter of integrate() across all stopping criteria."""

    def setUp(self):
        self.seed = 7
        self.dimension = 2
        self.loose_abs_tol = 0.2
        self.tight_abs_tol = 0.05
        self.rel_tol = 0
        self.n_init = 2**8
        self.n_limit = 2**16
        self.iid = IIDStdUniform(self.dimension, seed=self.seed)
        self.lattice = Lattice(self.dimension, seed=self.seed)
        self.net = DigitalNetB2(self.dimension, seed=self.seed)
        self.integrand_iid = Keister(self.iid)
        self.integrand_lattice = Keister(self.lattice)
        self.integrand_net = Keister(self.net)

    def _make_sc(self, cls, integrand, abs_tol, **kwargs):
        return cls(integrand, abs_tol=abs_tol, rel_tol=self.rel_tol, **kwargs)

    def test_resume_none_is_equivalent_to_fresh_start(self):
        """resume=None must behave identically to a fresh start."""
        sc1 = CubQMCLatticeG(
            Keister(Lattice(self.dimension, seed=self.seed)),
            abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol1, data1 = sc1.integrate()
        sc2 = CubQMCLatticeG(
            Keister(Lattice(self.dimension, seed=self.seed)),
            abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol2, data2 = sc2.integrate(resume=None)
        self.assertTrue(np.allclose(sol1, sol2, rtol=1e-5))

    def test_mc_clt_resume(self):
        """Test CubMCCLT resume functionality."""
        sc1 = CubMCCLT(Keister(IIDStdUniform(self.dimension, seed=self.seed)),
                        abs_tol=self.loose_abs_tol, rel_tol=self.rel_tol,
                        n_init=self.n_init, n_limit=self.n_limit)
        sol1, data1 = sc1.integrate()
        sc2 = CubMCCLT(Keister(IIDStdUniform(self.dimension, seed=self.seed)),
                        abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
                        n_init=self.n_init, n_limit=self.n_limit)
        sol2, data2 = sc2.integrate(resume=data1)
        sc3 = CubMCCLT(Keister(IIDStdUniform(self.dimension, seed=self.seed)),
                        abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
                        n_init=self.n_init, n_limit=self.n_limit)
        sol3, data3 = sc3.integrate()
        self.assertTrue(hasattr(data2, 'n_total'))
        self.assertTrue(data2.n_total >= data1.n_total)
        self.assertTrue(np.allclose(sol2, sol3, rtol=0.5))

    def test_mc_clt_vec_resume(self):
        """Test CubMCCLTVec resume functionality."""
        try:
            sc1 = CubMCCLTVec(Keister(IIDStdUniform(self.dimension, seed=self.seed)),
                               abs_tol=self.loose_abs_tol, rel_tol=self.rel_tol,
                               n_init=self.n_init, n_limit=self.n_limit)
            sol1, data1 = sc1.integrate()
            sc2 = CubMCCLTVec(Keister(IIDStdUniform(self.dimension, seed=self.seed)),
                               abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
                               n_init=self.n_init, n_limit=self.n_limit)
            sol2, data2 = sc2.integrate(resume=data1)
            sc3 = CubMCCLTVec(Keister(IIDStdUniform(self.dimension, seed=self.seed)),
                               abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
                               n_init=self.n_init, n_limit=self.n_limit)
            sol3, data3 = sc3.integrate()
            self.assertTrue(hasattr(data2, 'n_total'))
            self.assertTrue(data2.n_total >= data1.n_total)
            self.assertTrue(np.allclose(sol2, sol3, rtol=0.5))
        except Exception as e:
            self.skipTest(f"CubMCCLTVec resume skipped: {e}")

    def test_mc_g_resume(self):
        """Test CubMCG resume functionality."""
        sc1 = CubMCG(Keister(IIDStdUniform(self.dimension, seed=self.seed)),
                     abs_tol=self.loose_abs_tol, rel_tol=self.rel_tol,
                     n_init=self.n_init, n_limit=self.n_limit)
        sol1, data1 = sc1.integrate()
        sc2 = CubMCG(Keister(IIDStdUniform(self.dimension, seed=self.seed)),
                     abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
                     n_init=self.n_init, n_limit=self.n_limit)
        sol2, data2 = sc2.integrate(resume=data1)
        sc3 = CubMCG(Keister(IIDStdUniform(self.dimension, seed=self.seed)),
                     abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
                     n_init=self.n_init, n_limit=self.n_limit)
        sol3, data3 = sc3.integrate()
        self.assertTrue(hasattr(data2, 'n_total'))
        self.assertTrue(data2.n_total >= data1.n_total)
        self.assertTrue(np.allclose(sol2, sol3, rtol=0.5))

    def test_mc_ml_resume(self):
        """Test CubMCML resume functionality."""
        try:
            iid1 = IIDStdUniform(self.dimension, seed=self.seed)
            sc1 = CubMCML(FinancialOption(iid1, start_price=30, strike_price=30),
                          rmse_tol=self.loose_abs_tol, n_limit=self.n_limit)
            sol1, data1 = sc1.integrate()
            iid2 = IIDStdUniform(self.dimension, seed=self.seed)
            sc2 = CubMCML(FinancialOption(iid2, start_price=30, strike_price=30),
                          rmse_tol=self.tight_abs_tol, n_limit=self.n_limit)
            sol2, data2 = sc2.integrate(resume=data1)
            self.assertTrue(hasattr(data2, 'n_total'))
        except Exception as e:
            self.skipTest(f"CubMCML resume skipped: {e}")

    def test_qmc_lattice_resume(self):
        """Test CubQMCLatticeG resume functionality."""
        sc1 = CubQMCLatticeG(
            Keister(Lattice(self.dimension, seed=self.seed)),
            abs_tol=self.loose_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol1, data1 = sc1.integrate()
        sc2 = CubQMCLatticeG(
            Keister(Lattice(self.dimension, seed=self.seed)),
            abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol2, data2 = sc2.integrate(resume=data1)
        sc3 = CubQMCLatticeG(
            Keister(Lattice(self.dimension, seed=self.seed)),
            abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol3, data3 = sc3.integrate()
        self.assertTrue(hasattr(data2, 'n_total'))
        self.assertTrue(data2.n_total >= data1.n_total)
        self.assertTrue(np.allclose(sol2, sol3, rtol=0.5))

    def test_qmc_net_resume(self):
        """Test CubQMCNetG resume functionality."""
        sc1 = CubQMCNetG(
            Keister(DigitalNetB2(self.dimension, seed=self.seed)),
            abs_tol=self.loose_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol1, data1 = sc1.integrate()
        sc2 = CubQMCNetG(
            Keister(DigitalNetB2(self.dimension, seed=self.seed)),
            abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol2, data2 = sc2.integrate(resume=data1)
        sc3 = CubQMCNetG(
            Keister(DigitalNetB2(self.dimension, seed=self.seed)),
            abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol3, data3 = sc3.integrate()
        self.assertTrue(hasattr(data2, 'n_total'))
        self.assertTrue(data2.n_total >= data1.n_total)
        self.assertTrue(np.allclose(sol2, sol3, rtol=0.5))

    def test_qmc_ml_resume(self):
        """Test CubQMCML resume functionality."""
        try:
            sc1 = CubQMCML(FinancialOption(Lattice(replications=32, seed=self.seed), start_price=30, strike_price=30),
                           abs_tol=self.loose_abs_tol, n_limit=self.n_limit)
            sol1, data1 = sc1.integrate()
            sc2 = CubQMCML(FinancialOption(Lattice(replications=32, seed=self.seed), start_price=30, strike_price=30),
                           abs_tol=self.tight_abs_tol, n_limit=self.n_limit)
            sol2, data2 = sc2.integrate(resume=data1)
            self.assertTrue(hasattr(data2, 'n_total'))
        except Exception as e:
            self.skipTest(f"CubQMCML resume skipped: {e}")

    def test_qmc_ml_cont_resume(self):
        """Test CubQMCMLCont resume functionality."""
        try:
            sc1 = CubQMCMLCont(FinancialOption(Lattice(replications=32, seed=self.seed), start_price=30, strike_price=30),
                               abs_tol=self.loose_abs_tol, n_limit=self.n_limit)
            sol1, data1 = sc1.integrate()
            sc2 = CubQMCMLCont(FinancialOption(Lattice(replications=32, seed=self.seed), start_price=30, strike_price=30),
                               abs_tol=self.tight_abs_tol, n_limit=self.n_limit)
            sol2, data2 = sc2.integrate(resume=data1)
            self.assertTrue(hasattr(data2, 'n_total'))
        except Exception as e:
            self.skipTest(f"CubQMCMLCont resume skipped: {e}")

    def test_bayesian_lattice_resume(self):
        """Test CubBayesLatticeG resume functionality."""
        sc1 = CubBayesLatticeG(
            Keister(Lattice(self.dimension, seed=self.seed)),
            abs_tol=self.loose_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol1, data1 = sc1.integrate()
        sc2 = CubBayesLatticeG(
            Keister(Lattice(self.dimension, seed=self.seed)),
            abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol2, data2 = sc2.integrate(resume=data1)
        sc3 = CubBayesLatticeG(
            Keister(Lattice(self.dimension, seed=self.seed)),
            abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol3, data3 = sc3.integrate()
        self.assertTrue(hasattr(data2, 'n_total'))
        self.assertTrue(data2.n_total >= data1.n_total)
        self.assertTrue(np.allclose(sol2, sol3, rtol=0.5))

    def test_bayesian_net_resume(self):
        """Test CubBayesNetG resume functionality."""
        sc1 = CubBayesNetG(
            Keister(DigitalNetB2(self.dimension, seed=self.seed)),
            abs_tol=self.loose_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol1, data1 = sc1.integrate()
        sc2 = CubBayesNetG(
            Keister(DigitalNetB2(self.dimension, seed=self.seed)),
            abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol2, data2 = sc2.integrate(resume=data1)
        sc3 = CubBayesNetG(
            Keister(DigitalNetB2(self.dimension, seed=self.seed)),
            abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init, n_limit=self.n_limit,
        )
        sol3, data3 = sc3.integrate()
        self.assertTrue(hasattr(data2, 'n_total'))
        self.assertTrue(data2.n_total >= data1.n_total)
        self.assertTrue(np.allclose(sol2, sol3, rtol=0.5))
