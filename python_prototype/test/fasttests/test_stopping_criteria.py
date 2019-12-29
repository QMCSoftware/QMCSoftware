""" Unit tests for subclasses of StoppingCriterion in QMCPy """

from qmcpy import *
from qmcpy._util import *

from numpy import arange
import unittest
import warnings

tv_single_level = [arange(1 / 64, 65 / 64, 1 / 64)]
dim_single_level = [len(tv) for tv in tv_single_level]

tv_multi_level = [arange(1 / 4, 5 / 4, 1 / 4),
                  arange(1 / 16, 17 / 16, 1 / 16),
                  arange(1 / 64, 65 / 64, 1 / 64)]
dim_multi_level = [len(tv) for tv in tv_multi_level]


class TestClt(unittest.TestCase):
    """
    Unit tests for Clt in QMCPy.
    """

    def test_raise_distribution_compatibility_error(self):
        self.assertRaises(DistributionCompatibilityError, CLT, Lattice(),
                          Gaussian(3))

    def test_n_max_single_level(self):
        discrete_distrib = IIDStdUniform(rng_seed=7)
        true_measure = BrownianMotion(dim_single_level,
                                      time_vector=tv_single_level)
        integrand = AsianCall(true_measure)
        stopping_criterion = CLT(discrete_distrib, true_measure,
                                 abs_tol=.1, n_init=64, n_max=1000)
        self.assertWarns(MaxSamplesWarning, integrate, integrand,
                         true_measure, discrete_distrib, stopping_criterion)

    def test_n_max_multi_level(self):
        discrete_distrib = IIDStdUniform(rng_seed=7)
        true_measure = BrownianMotion(dim_multi_level,
                                      time_vector=tv_multi_level)
        integrand = AsianCall(true_measure)
        stopping_criterion = CLT(discrete_distrib, true_measure,
                                 abs_tol=.1, n_init=64, n_max=1000)
        self.assertWarns(MaxSamplesWarning, integrate, integrand,
                         true_measure, discrete_distrib, stopping_criterion)


class TestCltRep(unittest.TestCase):
    """
    Unit tests for CltRep in QMCPy.
    """

    def test_raise_distribution_compatibility_error(self):
        self.assertRaises(DistributionCompatibilityError, CLTRep,
                          IIDStdGaussian(), Gaussian(3))

    def test_n_init_power_of_2(self):
        self.assertWarns(ParameterWarning, CLTRep,
                         Lattice(), Gaussian(3), n_init=45)

    def test_n_max_single_level(self):
        discrete_distrib = Lattice(rng_seed=7)
        true_measure = BrownianMotion(dim_single_level,
                                      time_vector=tv_single_level)
        integrand = AsianCall(true_measure)
        stopping_criterion = CLTRep(discrete_distrib, true_measure,
                                    abs_tol=.1, n_init=32, n_max=100)
        self.assertWarns(MaxSamplesWarning, integrate, integrand,
                         true_measure, discrete_distrib, stopping_criterion)


class TestMeanMC_g(unittest.TestCase):
    """
    Unit tests for MeanMC_g in QMCPy
    """

    def test_raise_distribution_compatibility_error(self):
        self.assertRaises(DistributionCompatibilityError, CLT, Lattice(),
                          Gaussian(3))

    def test_n_max_single_level(self):
        discrete_distrib = IIDStdUniform(rng_seed=7)
        true_measure = BrownianMotion(dim_single_level,
                                      time_vector=tv_single_level)
        integrand = AsianCall(true_measure)
        stopping_criterion = CLT(discrete_distrib, true_measure,
                                 abs_tol=.1, n_init=64, n_max=1000)
        self.assertWarns(MaxSamplesWarning, integrate, integrand,
                         true_measure, discrete_distrib, stopping_criterion)


if __name__ == "__main__":
    unittest.main()
