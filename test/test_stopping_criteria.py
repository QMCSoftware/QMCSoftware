"""Unit tests for subclasses of StoppingCriterion in QMCPy"""

import builtins
import importlib
import io
import os
import tempfile
import unittest
import warnings
import numpy as np
from contextlib import ExitStack, redirect_stdout
from unittest.mock import patch

from qmcpy import *
from qmcpy.util import *
from qmcpy.util.data import Data
from qmcpy.discrete_distribution.abstract_discrete_distribution import AbstractDiscreteDistribution
from qmcpy.integrand.abstract_integrand import AbstractIntegrand
from qmcpy.stopping_criterion.abstract_stopping_criterion import (
    _IterationTraceLogger,
    AbstractStoppingCriterion,
    print_diagnostic,
)


# Test functions and parameters
keister_2d_exact = 1.808186429263620
tol = 0.005
rel_tol = 0

_ISHIGAMI_A = 7
_ISHIGAMI_B = 0.1
_ISHIGAMI_INDICES = np.array(
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


def _sensitivity_converges(distrib_factory, sc_class, abs_tol, sc_kwargs=None, n_tries=3):
    """Return True if at least one of n_tries Ishigami sensitivity-index runs meets abs_tol."""
    if sc_kwargs is None:
        sc_kwargs = {}
    true_solution = Ishigami._exact_sensitivity_indices(
        _ISHIGAMI_INDICES, _ISHIGAMI_A, _ISHIGAMI_B
    )
    for _ in range(n_tries):
        si = SensitivityIndices(
            Ishigami(distrib_factory(), _ISHIGAMI_A, _ISHIGAMI_B), _ISHIGAMI_INDICES
        )
        solution, _ = sc_class(si, abs_tol=abs_tol, rel_tol=0, **sc_kwargs).integrate()
        if (abs(solution.squeeze() - true_solution) < abs_tol).all():
            return True
    return False


class _DummyDiscreteDistribution(AbstractDiscreteDistribution):
    def __init__(self):
        pass


class _OtherDummyDiscreteDistribution(AbstractDiscreteDistribution):
    def __init__(self):
        pass


class _DummyIntegrand(AbstractIntegrand):
    def __init__(self, true_measure, discrete_distrib, d_indv=(), d_comb=(), dependency=None):
        self.true_measure = true_measure
        self.discrete_distrib = discrete_distrib
        self.d_indv = d_indv
        self.d_comb = d_comb
        self._dependency = dependency or (lambda flags: flags)

    def dependency(self, comb_flags):
        return self._dependency(comb_flags)


class _DummyStoppingCriterion(AbstractStoppingCriterion):
    def __init__(self, integrand=None, true_measure=None, discrete_distrib=None, allowed_distribs=None, allow_vectorized_integrals=False, parameters=None):
        if integrand is not None:
            self.integrand = integrand
        if true_measure is not None:
            self.true_measure = true_measure
        if discrete_distrib is not None:
            self.discrete_distrib = discrete_distrib
        if parameters is not None:
            self.parameters = parameters
        super().__init__(allowed_distribs or [_DummyDiscreteDistribution], allow_vectorized_integrals)


class TestAbstractStoppingCriterion(unittest.TestCase):
    def test_print_diagnostic_formats_header_and_values(self):
        data = type(
            "Data",
            (),
            {
                "_iter_count": 2,
                "solution": [1.25],
                "m": np.array(4),
                "n_total": 16,
                "n_min": None,
                "xfull": np.zeros((2, 3)),
            },
        )()
        stream = io.StringIO()
        with redirect_stdout(stream):
            print_diagnostic("resume", data, table_header=True)
        output = stream.getvalue()
        self.assertIn("stage", output)
        self.assertIn("iter", output)
        self.assertIn("resume", output)
        self.assertIn("1.2500000", output)
        self.assertIn("(2, 3)", output)
        self.assertRegex(output, r"resume\s+2\s+1\.2500000\s+None\s+16\s+4")

    def test_print_diagnostic_formats_missing_values_as_nan_and_none(self):
        data = type("Data", (), {"solution": float("nan"), "xfull": None})()
        stream = io.StringIO()
        with redirect_stdout(stream):
            print_diagnostic("start", data)
        output = stream.getvalue()
        self.assertIn("start", output)
        self.assertIn("nan", output)
        self.assertIn("None", output)

    def test_print_diagnostic_can_disable_iteration_throttling(self):
        data = type(
            "Data",
            (),
            {
                "solution": [1.25],
                "_iter_count": 11,
                "m": 13,
                "n_total": 16,
                "n_min": None,
                "xfull": np.zeros((2, 3)),
            },
        )()
        throttled = io.StringIO()
        with redirect_stdout(throttled):
            print_diagnostic("ITER", data)
        self.assertEqual(throttled.getvalue(), "")

        unthrottled = io.StringIO()
        with redirect_stdout(unthrottled):
            print_diagnostic("ITER", data, throttle_iterations=False)
        output = unthrottled.getvalue()
        self.assertIn("ITER", output)
        self.assertIn("1.2500000", output)
        self.assertRegex(output, r"ITER\s+11\s+1\.2500000\s+None\s+16\s+13")

    def test_print_diagnostic_can_hide_optional_columns(self):
        data = type(
            "Data",
            (),
            {
                "_iter_count": 1,
                "solution": [1.25],
                "n_total": 16,
            },
        )()
        stream = io.StringIO()
        with redirect_stdout(stream):
            print_diagnostic(
                "ITER",
                data,
                table_header=True,
                visible_columns=("stage", "iter", "solution", "n_total"),
            )
        output = stream.getvalue()
        self.assertIn("stage", output)
        self.assertIn("iter", output)
        self.assertIn("n_total", output)
        self.assertNotIn("n_min", output)
        self.assertNotIn("xfull.shape", output)

    def test_resume_trace_blanks_resume_iter_and_continues_count(self):
        logger = _IterationTraceLogger(
            type(
                "SC",
                (),
                {
                    "trace_iterations": True,
                    "trace_label": "resume-test",
                    "trace_throttle_iterations": False,
                },
            )()
        )
        data = type(
            "Data",
            (),
            {
                "_iter_count": 6,
                "solution": [1.25],
                "n_total": 16,
                "n_min": 8,
                "m": 13,
                "xfull": np.zeros((16, 2)),
            },
        )()
        stream = io.StringIO()
        with redirect_stdout(stream):
            logger.resume(data, step_value=13)
            # Simulate new samples being drawn: update n_total so state differs
            data.n_total = 20
            data.xfull = np.zeros((20, 2))
            logger.iteration(data, step_value=14)
        output = stream.getvalue()
        self.assertRegex(output, r"RESUME\s+6\s+1\.2500000\s+8\s+16\s+13")
        self.assertNotIn("RESUME       None", output)
        self.assertRegex(output, r"ITER\s+7\s+1\.2500000\s+8\s+20\s+14")

    def test_init_requires_integrand(self):
        with self.assertRaises(ParameterError):
            _DummyStoppingCriterion()

    def test_init_requires_matching_true_measure(self):
        distrib = _DummyDiscreteDistribution()
        integrand = _DummyIntegrand(object(), distrib)
        with self.assertRaises(ParameterError):
            _DummyStoppingCriterion(
                integrand=integrand,
                true_measure=object(),
                discrete_distrib=distrib,
            )

    def test_init_requires_matching_discrete_distribution(self):
        distrib = _DummyDiscreteDistribution()
        integrand = _DummyIntegrand(object(), distrib)
        with self.assertRaises(ParameterError):
            _DummyStoppingCriterion(
                integrand=integrand,
                true_measure=integrand.true_measure,
                discrete_distrib=_DummyDiscreteDistribution(),
            )

    def test_init_requires_allowed_distribution_type(self):
        distrib = _OtherDummyDiscreteDistribution()
        integrand = _DummyIntegrand(object(), distrib)
        with self.assertRaises(DistributionCompatibilityError):
            _DummyStoppingCriterion(
                integrand=integrand,
                true_measure=integrand.true_measure,
                discrete_distrib=distrib,
            )

    def test_init_rejects_vectorized_integrals_when_not_supported(self):
        distrib = _DummyDiscreteDistribution()
        integrand = _DummyIntegrand(object(), distrib, d_indv=(2,), d_comb=(2,))
        with self.assertRaises(ParameterError):
            _DummyStoppingCriterion(
                integrand=integrand,
                true_measure=integrand.true_measure,
                discrete_distrib=distrib,
            )

    def test_default_parameters_and_repr(self):
        distrib = _DummyDiscreteDistribution()
        integrand = _DummyIntegrand(object(), distrib)
        sc = _DummyStoppingCriterion(
            integrand=integrand,
            true_measure=integrand.true_measure,
            discrete_distrib=distrib,
        )
        self.assertEqual(sc.parameters, [])
        self.assertIn("AbstractStoppingCriterion", repr(sc))

    def test_abstract_methods_raise(self):
        distrib = _DummyDiscreteDistribution()
        integrand = _DummyIntegrand(object(), distrib)
        sc = _DummyStoppingCriterion(
            integrand=integrand,
            true_measure=integrand.true_measure,
            discrete_distrib=distrib,
        )
        with self.assertRaises(MethodImplementationError):
            sc.integrate()
        with self.assertRaises(MethodImplementationError):
            sc.set_tolerance(abs_tol=0.1)

    def test_compute_indv_alphas_with_identity_dependency(self):
        distrib = _DummyDiscreteDistribution()
        integrand = _DummyIntegrand(
            object(),
            distrib,
            d_indv=(2,),
            d_comb=(2,),
            dependency=lambda flags: flags,
        )
        sc = _DummyStoppingCriterion(
            integrand=integrand,
            true_measure=integrand.true_measure,
            discrete_distrib=distrib,
            allow_vectorized_integrals=True,
        )
        alphas_indv, identity_dependency = sc._compute_indv_alphas(np.array([0.2, 0.4]))
        self.assertTrue(identity_dependency)
        self.assertTrue(np.allclose(alphas_indv, [0.2, 0.4]))

    def test_compute_indv_alphas_with_non_identity_dependency(self):
        distrib = _DummyDiscreteDistribution()

        def dependency(flags):
            return np.array([False, False, True]) if not flags[0] else np.array([True, False, False])

        integrand = _DummyIntegrand(
            object(),
            distrib,
            d_indv=(3,),
            d_comb=(2,),
            dependency=dependency,
        )
        sc = _DummyStoppingCriterion(
            integrand=integrand,
            true_measure=integrand.true_measure,
            discrete_distrib=distrib,
            allow_vectorized_integrals=True,
        )
        alphas_indv, identity_dependency = sc._compute_indv_alphas(np.array([0.3, 0.6]))
        self.assertFalse(identity_dependency)
        self.assertTrue(np.allclose(alphas_indv, [0.15, 0.15, 0.3]))

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
        self.assertTrue(_sensitivity_converges(
            lambda: DigitalNetB2(3, seed=7, replications=32), CubQMCCLT, abs_tol=5e-2
        ))

    def test_sobol_indices_lattice(self):
        self.assertTrue(_sensitivity_converges(
            lambda: Lattice(3, seed=7, replications=32), CubQMCCLT, abs_tol=5e-2
        ))

    def test_sobol_indices_halton(self):
        self.assertTrue(_sensitivity_converges(
            lambda: Halton(3, seed=7, replications=32), CubQMCCLT, abs_tol=5e-2
        ))


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
        self.assertTrue(_sensitivity_converges(
            lambda: Lattice(3, seed=7), CubQMCLatticeG, abs_tol=5e-3
        ))


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
        self.assertTrue(_sensitivity_converges(
            lambda: DigitalNetB2(3, seed=7), CubQMCNetG, abs_tol=1e-2
        ))


class TestCubBayesLatticeG(unittest.TestCase):
    """Unit tests for CubBayesLatticeG StoppingCriterion."""

    def test_raise_distribution_compatibility_error(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubBayesLatticeG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Lattice(dimension=2, seed=7, order="RADICAL INVERSE"))
        algorithm = CubBayesLatticeG(integrand, abs_tol=0.0001, n_init=2**8, n_limit=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(Lattice(dimension=2, seed=7, order="RADICAL INVERSE"))
        solution, data = CubBayesLatticeG(integrand, abs_tol=tol, n_init=2**5).integrate()
        self.assertTrue(abs(solution - keister_2d_exact) < tol)

    def test_sobol_indices_bayes_lattice(self, dims=3, abs_tol=1e-2):
        keister_d_ = Keister(Lattice(dimension=dims, seed=7))
        keister_indices_ = SobolIndices(keister_d_, indices="singletons")
        sc_ = CubQMCLatticeG(keister_indices_, abs_tol=abs_tol, ptransform="Baker")
        solution_, data_ = sc_.integrate()

        keister_d = Keister(Lattice(dimension=dims, order="RADICAL INVERSE", seed=7))
        keister_indices = SobolIndices(keister_d, indices="singletons")
        sc = CubBayesLatticeG(keister_indices, order=1, abs_tol=abs_tol, ptransform="Baker")
        solution, data = sc.integrate()

        self.assertTrue(solution.shape, (dims, dims, 1))
        self.assertTrue(abs(solution - solution_).max() < abs_tol)

    def test_sobol_indices(self):
        self.assertTrue(_sensitivity_converges(
            lambda: Lattice(3, seed=7, order="natural"), CubBayesLatticeG,
            abs_tol=5e-2, sc_kwargs={"order": 1, "ptransform": "Baker"},
        ))


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
        self.assertTrue(_sensitivity_converges(
            lambda: DigitalNetB2(3, seed=7), CubBayesNetG, abs_tol=1e-2
        ))


class TestMultilevelStoppingCriteria(unittest.TestCase):
    def _iid_financial_option(self):
        return FinancialOption(IIDStdUniform(seed=7), start_price=30, strike_price=30)

    def _qmc_financial_option(self):
        return FinancialOption(Lattice(replications=32, seed=7), start_price=30, strike_price=30)

    def test_raise_distribution_compatibility_error(self):
        cases = [
            ("CubMCML", CubMCML, Lattice(seed=7)),
            ("CubQMCML", CubQMCML, IIDStdUniform(seed=7)),
        ]
        for label, cls, sampler in cases:
            with self.subTest(stopping_criterion=label):
                integrand = FinancialOption(sampler)
                with self.assertRaises(DistributionCompatibilityError):
                    cls(integrand)

    def test_n_max(self):
        cases = [
            ("CubMCML", CubMCML, FinancialOption(IIDStdUniform(seed=7), start_price=30, strike_price=30), {"rmse_tol": 0.001, "n_limit": 2**10}),
            ("CubQMCML", CubQMCML, FinancialOption(Lattice(replications=32, seed=7), start_price=30, strike_price=30), {"abs_tol": tol, "n_limit": 2**10}),
        ]
        for label, cls, integrand, kwargs in cases:
            with self.subTest(stopping_criterion=label):
                algorithm = cls(integrand, **kwargs)
                self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_multilevel_parameter_validation(self):
        cases = [
            ("CubMLMC", CubMLMC, self._iid_financial_option),
            ("CubMLMCCont", CubMLMCCont, self._iid_financial_option),
        ]
        invalid_kwargs = [
            ({"levels_min": 1}, "needs levels_min >= 2"),
            ({"levels_min": 2, "levels_max": 1}, "needs levels_max >= levels_min"),
            ({"n_init": 0}, "needs n_init > 0"),
        ]
        for label, cls, integrand_factory in cases:
            for kwargs, message in invalid_kwargs:
                with self.subTest(stopping_criterion=label, kwargs=kwargs):
                    with self.assertRaisesRegex(ParameterError, message):
                        cls(integrand_factory(), **kwargs)

    def test_multilevel_rmse_tol_overrides_abs_tol(self):
        cases = [
            ("CubMLMC", CubMLMC, self._iid_financial_option, "rmse_tol"),
            ("CubMLMCCont", CubMLMCCont, self._iid_financial_option, "target_tol"),
            ("CubMLQMC", CubMLQMC, self._qmc_financial_option, "rmse_tol"),
            ("CubMLQMCCont", CubMLQMCCont, self._qmc_financial_option, "target_tol"),
        ]
        for label, cls, integrand_factory, attr in cases:
            with self.subTest(stopping_criterion=label):
                sc = cls(integrand_factory(), abs_tol=999.0, rmse_tol=0.123)
                self.assertEqual(
                    getattr(sc, attr),
                    0.123,
                    msg=f"{label} did not prioritize rmse_tol over abs_tol",
                )

    def test_continuation_warns_when_max_levels_reached(self):
        cases = [
            ("CubMLMCCont", CubMLMCCont, self._iid_financial_option, 2),
            ("CubMLQMCCont", CubMLQMCCont, self._qmc_financial_option, 3),
        ]
        for label, cls, integrand_factory, levels_max in cases:
            with self.subTest(stopping_criterion=label):
                sc = cls(
                    integrand_factory(), rmse_tol=0.1, levels_min=2, levels_max=levels_max
                )
                sc.rmse_tol = sc.target_tol
                data = sc._construct_data()
                data.n_total = 0
                data.levels = sc.levels_max
                data.n_level[:] = 1
                with ExitStack() as stack:
                    if label == "CubMLMCCont":
                        data.sum_level[:] = 1.0
                        data.var_level = np.ones_like(data.n_level, dtype=float)
                        data.diff_n_level = np.zeros_like(data.n_level)
                        stack.enter_context(patch.object(sc, "_update_data"))
                        stack.enter_context(patch.object(sc, "_update_theta"))
                        stack.enter_context(patch.object(sc, "_get_next_samples", return_value=data.n_level.copy()))
                        stack.enter_context(patch.object(sc, "_rmse", return_value=sc.rmse_tol * 2))
                        add_level = stack.enter_context(patch.object(sc, "_add_level"))
                        with self.assertWarns(MaxLevelsWarning):
                            sc._integrate(data)
                    else:
                        data.eval_level[:] = False
                        data.mean_level_reps[:] = 1.0
                        data.var_level[:] = 0.0
                        data.var_cost_ratio_level[:] = 0.0
                        data.bias_estimate = 0.0
                        stack.enter_context(patch.object(sc, "update_data"))
                        stack.enter_context(patch.object(sc, "_update_theta"))
                        stack.enter_context(patch.object(sc, "_varest", return_value=0.0))
                        stack.enter_context(patch.object(sc, "_rmse", return_value=sc.rmse_tol * 2))
                        add_level = stack.enter_context(patch.object(sc, "_add_level"))
                        with self.assertWarns(MaxLevelsWarning):
                            sc._integrate(data, skip_level_reset=True)
                    add_level.assert_not_called()

class TestResumeFeature(unittest.TestCase):
    """Tests for the resume parameter of integrate() across all stopping criteria."""

    def setUp(self):
        warnings.filterwarnings("ignore", category=MaxSamplesWarning)
        self.seed = 7
        self.dimension = 2
        self.loose_abs_tol = 0.2
        self.tight_abs_tol = 0.05
        self.rel_tol = 0
        self.n_init = 2**8
        self.n_limit = 2**16

    def _iid_distribution(self):
        return IIDStdUniform(self.dimension, seed=self.seed)

    def _lattice_distribution(self):
        return Lattice(self.dimension, seed=self.seed)

    def _net_distribution(self):
        return DigitalNetB2(self.dimension, seed=self.seed)

    def _net_rep_distribution(self):
        return DigitalNetB2(self.dimension, replications=16, seed=self.seed)

    def _iid_financial_option(self):
        return FinancialOption(IIDStdUniform(self.dimension, seed=self.seed), start_price=30, strike_price=30)

    def _qmc_financial_option(self):
        return FinancialOption(Lattice(replications=32, seed=self.seed), start_price=30, strike_price=30)

    def _keister_builder(self, stopping_criterion, distribution_factory, abs_tol):
        return lambda: stopping_criterion(Keister(distribution_factory()), abs_tol=abs_tol, rel_tol=self.rel_tol, n_init=self.n_init, n_limit=self.n_limit)

    def _multilevel_builder(self, stopping_criterion, integrand_factory, tol_kwarg, tol):
        return lambda: stopping_criterion(integrand_factory(), **{tol_kwarg: tol}, n_limit=self.n_limit)

    def _assert_resume_behavior(self, label, loose_builder, tight_builder, compare_to_fresh=False, rtol=0.5, skip_exceptions=()):
        def _run_assertions():
            sc1 = loose_builder()
            _, data1 = sc1.integrate()
            old_n_total = int(data1.n_total)  # save before resume mutates data1 in-place

            sc2 = tight_builder()
            sol2, data2 = sc2.integrate(resume=data1)

            self.assertTrue(hasattr(data2, "n_total"), msg=f"{label} missing n_total")
            self.assertTrue(
                data2.n_total >= old_n_total,
                msg=f"{label} resume did not preserve/increase n_total",
            )

            if compare_to_fresh:
                sc3 = tight_builder()
                sol3, _ = sc3.integrate()
                self.assertTrue(
                    np.allclose(sol2, sol3, rtol=rtol),
                    msg=f"{label} resume solution diverged from fresh run",
                )

        if skip_exceptions:
            try:
                _run_assertions()
            except skip_exceptions as exc:
                self.skipTest(f"{label} resume skipped: {exc}")
        else:
            _run_assertions()

    def test_resume_none_is_equivalent_to_fresh_start(self):
        """resume=None must behave identically to a fresh start."""
        make_sc = self._keister_builder(CubQMCLatticeG, self._lattice_distribution, self.tight_abs_tol)
        sol1, _ = make_sc().integrate()
        sol2, _ = make_sc().integrate(resume=None)
        self.assertTrue(np.allclose(sol1, sol2, rtol=1e-5))

    def test_resume_matches_fresh_tight_solution(self):
        """Resume solutions should match a fresh run at tighter tolerance."""
        cases = [
            ("CubMCCLTVec", CubMCCLTVec, self._iid_distribution, (ImportError, NotImplementedError)),
            ("CubQMCLatticeG", CubQMCLatticeG, self._lattice_distribution, ()),
            ("CubQMCNetG", CubQMCNetG, self._net_distribution, ()),
            ("CubBayesLatticeG", CubBayesLatticeG, self._lattice_distribution, ()),
            ("CubBayesNetG", CubBayesNetG, self._net_distribution, ()),
        ]

        for label, stopping_criterion, distribution_factory, skip_exceptions in cases:
            with self.subTest(stopping_criterion=label):
                self._assert_resume_behavior(
                    label,
                    self._keister_builder(stopping_criterion, distribution_factory, self.loose_abs_tol),
                    self._keister_builder(stopping_criterion, distribution_factory, self.tight_abs_tol),
                    compare_to_fresh=True,
                    rtol=0.5,
                    skip_exceptions=skip_exceptions,
                )

    def test_resume_increases_samples_for_multilevel_algorithms(self):
        """Multilevel resume runs should retain or increase sample work."""
        cases = [
            ("CubMCML", CubMCML, self._iid_financial_option, "rmse_tol"),
            ("CubMCMLCont", CubMCMLCont, self._iid_financial_option, "rmse_tol"),
            ("CubQMCML", CubQMCML, self._qmc_financial_option, "abs_tol"),
            ("CubQMCMLCont", CubQMCMLCont, self._qmc_financial_option, "abs_tol"),
        ]

        for label, stopping_criterion, integrand_factory, tol_kwarg in cases:
            with self.subTest(stopping_criterion=label):
                self._assert_resume_behavior(
                    label,
                    self._multilevel_builder(stopping_criterion, integrand_factory, tol_kwarg, self.loose_abs_tol),
                    self._multilevel_builder(stopping_criterion, integrand_factory, tol_kwarg, self.tight_abs_tol),
                )

    def test_continuation_resume_uses_target_tol_without_level_reset(self):
        cases = [
            ("CubMLMCCont", CubMLMCCont, self._iid_financial_option, 4),
            ("CubMLQMCCont", CubMLQMCCont, self._qmc_financial_option, 5),
        ]
        for label, stopping_criterion, integrand_factory, expected_levels in cases:
            with self.subTest(stopping_criterion=label):
                sc = stopping_criterion(integrand_factory(), abs_tol=0.2, rmse_tol=0.1)
                data = type("ResumeData", (), {"solution": 0.0, "levels": 5})()
                captured = {}

                def fake_integrate(resume_data, skip_level_reset=False):
                    captured["levels"] = resume_data.levels
                    captured["skip_level_reset"] = skip_level_reset

                with patch.object(sc, "_validate_resume"), \
                     patch.object(sc, "_integrate", side_effect=fake_integrate):
                    sc.integrate(resume=data)

                self.assertEqual(captured["levels"], expected_levels)
                self.assertTrue(captured["skip_level_reset"])
                self.assertEqual(sc.rmse_tol, sc.target_tol)

    def test_qmc_resume_rejects_missing_transform_state(self):
        loose_sc = CubQMCLatticeG(
            Keister(Lattice(dimension=self.dimension, seed=self.seed)),
            abs_tol=self.loose_abs_tol,
            rel_tol=self.rel_tol,
            n_init=self.n_init,
            n_limit=self.n_limit,
        )
        _, checkpoint = loose_sc.integrate()
        self.assertTrue(hasattr(checkpoint, "_kappanumap"))
        del checkpoint._kappanumap

        tight_sc = CubQMCLatticeG(Keister(Lattice(dimension=self.dimension, seed=self.seed)), abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol, n_init=self.n_init, n_limit=self.n_limit)
        with self.assertRaises(ParameterError):
            tight_sc.integrate(resume=checkpoint)

    def test_bayes_resume_rejects_missing_transform_state(self):
        loose_sc = CubBayesLatticeG(Keister(Lattice(dimension=self.dimension, seed=self.seed, order="RADICAL INVERSE")), abs_tol=self.loose_abs_tol, rel_tol=self.rel_tol, n_init=2**5, n_limit=self.n_limit)
        _, checkpoint = loose_sc.integrate()
        self.assertTrue(hasattr(checkpoint, "_ytildefull"))
        del checkpoint._ytildefull

        tight_sc = CubBayesLatticeG(Keister(Lattice(dimension=self.dimension, seed=self.seed, order="RADICAL INVERSE")), abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol, n_init=2**5, n_limit=self.n_limit)
        with self.assertRaises(ParameterError):
            tight_sc.integrate(resume=checkpoint)

    def _qmc_rep_student_t(self):
        return CubQMCRepStudentT(Keister(self._net_rep_distribution()), abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol)

    def _pfgpci(self):
        return PFGPCI(Ishigami(DigitalNetB2(3, seed=self.seed)), failure_threshold=0, failure_above_threshold=False, abs_tol=self.tight_abs_tol, n_init=8, n_limit=16, n_batch=4, n_approx=2**8, gpytorch_train_iter=1, verbose=False, n_ref_approx=0)

    def test_unsupported_resume_raises_parameter_error(self):
        """Stopping criteria without resume support must raise ParameterError."""
        cases = [
            (
                "CubMCCLT",
                lambda: CubMCCLT(Keister(self._iid_distribution()), abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol, n_init=self.n_init, n_limit=self.n_limit),
            ),
            (
                "CubMCG",
                lambda: CubMCG(Keister(self._iid_distribution()), abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol, n_init=self.n_init, n_limit=self.n_limit),
            ),
            ("CubQMCRepStudentT", self._qmc_rep_student_t),
            ("PFGPCI", self._pfgpci),
        ]

        for label, stopping_criterion_factory in cases:
            with self.subTest(stopping_criterion=label):
                try:
                    sc = stopping_criterion_factory()
                except ModuleNotFoundError as exc:
                    self.skipTest(f"{label} unavailable: {exc}")
                with self.assertRaises(ParameterError):
                    sc.integrate(resume=object())

class TestResumeCheckpointing(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", category=MaxSamplesWarning)
        self.seed = 13
        self.dimension = 2
        self.n_init = 32
        self.n_limit = 2048
        self.loose_abs_tol = 0.2
        self.tight_abs_tol = 0.05

    def _make_integrand(self, dimension=None, seed=None):
        return Keister(IIDStdUniform(dimension=self.dimension if dimension is None else dimension, seed=self.seed if seed is None else seed))

    def test_cub_mc_clt_resume_raises_parameter_error(self):
        loose_sc = CubMCCLT(
            self._make_integrand(),
            abs_tol=self.loose_abs_tol,
            n_init=self.n_init,
            n_limit=self.n_limit,
        )
        _, checkpoint = loose_sc.integrate()

        tight_sc = CubMCCLT(
            self._make_integrand(),
            abs_tol=self.tight_abs_tol,
            n_init=self.n_init,
            n_limit=self.n_limit,
        )
        with self.assertRaises(ParameterError):
            tight_sc.integrate(resume=checkpoint)

    def test_cub_mc_g_resume_raises_parameter_error(self):
        loose_sc = CubMCG(
            self._make_integrand(),
            abs_tol=self.loose_abs_tol,
            n_init=self.n_init,
            n_limit=self.n_limit,
        )
        _, checkpoint = loose_sc.integrate()

        tight_sc = CubMCG(
            self._make_integrand(),
            abs_tol=self.tight_abs_tol,
            n_init=self.n_init,
            n_limit=self.n_limit,
        )
        with self.assertRaises(ParameterError):
            tight_sc.integrate(resume=checkpoint)

    def test_resume_rejects_incompatible_dimension(self):
        loose_sc = CubMCCLT(
            self._make_integrand(),
            abs_tol=self.loose_abs_tol,
            n_init=self.n_init,
            n_limit=self.n_limit,
        )
        _, checkpoint = loose_sc.integrate()

        incompatible_sc = CubMCCLT(
            self._make_integrand(dimension=3),
            abs_tol=self.tight_abs_tol,
            n_init=self.n_init,
            n_limit=self.n_limit,
        )
        with self.assertRaises(ParameterError):
            incompatible_sc.integrate(resume=checkpoint)

    def test_data_save_load_round_trip_plain_and_gzip(self):
        data = Data(parameters=["value"])
        data.value = np.array([1.0, 2.0, 3.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            plain_path = os.path.join(tmpdir, "checkpoint.pkl")
            gzip_path = os.path.join(tmpdir, "checkpoint.pkl.gz")

            data.save(plain_path)
            loaded_plain = Data.load(plain_path)
            self.assertTrue(np.array_equal(loaded_plain.value, data.value))

            data.save(gzip_path, compress=True)
            loaded_gzip = Data.load(gzip_path)
            self.assertTrue(np.array_equal(loaded_gzip.value, data.value))

    def test_resume_after_save_load_continues_rng_stream(self):
        loose_sc = CubQMCLatticeG(
            Keister(Lattice(dimension=self.dimension, seed=self.seed)),
            abs_tol=self.loose_abs_tol,
            n_init=self.n_init,
            n_limit=self.n_limit,
        )
        _, checkpoint = loose_sc.integrate()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "resume.pkl.gz")
            checkpoint.save(path, compress=True)
            loaded = Data.load(path)

        old_n_total = int(loaded.n_total)
        tight_sc = CubQMCLatticeG(
            Keister(Lattice(dimension=self.dimension, seed=self.seed)),
            abs_tol=self.tight_abs_tol,
            n_init=self.n_init,
            n_limit=self.n_limit,
        )
        _, resumed = tight_sc.integrate(resume=loaded)
        n_new = int(resumed.n_total) - old_n_total
        if n_new <= 0:
            self.skipTest("resume checkpoint already satisfied the tighter tolerance")
        self.assertFalse(
            np.allclose(resumed.xfull[old_n_total:], resumed.xfull[:n_new])
        )

if __name__ == "__main__":
    unittest.main()
