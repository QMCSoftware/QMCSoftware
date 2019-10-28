import unittest

from numpy import arange
from workouts.wo_abstol_runtime import comp_clt_vs_cltrep_runtimes
from workouts.wo_asian_option import test_distributions_asian_option
from workouts.wo_customizations import quick_construct_integrand
from workouts.wo_keister import test_distributions_keister


class TestWorkouts(unittest.TestCase):

    def test_abstol_runtime(self):
        comp_clt_vs_cltrep_runtimes(abstols=arange(.1, .4, .1))

    def test_asian_option(self):
        time_vec = [
            arange(1 / 4, 5 / 4, 1 / 4),
            arange(1 / 16, 17 / 16, 1 / 16),
            arange(1 / 64, 65 / 64, 1 / 64)]
        dim = [len(tv) for tv in time_vec]
        test_distributions_asian_option(time_vec, dim, abs_tol=.1)

    def test_keister(self):
        test_distributions_keister(dim=3, abs_tol=.1)

    def test_custom_customs(self):
        quick_construct_integrand(abs_tol=.1)
