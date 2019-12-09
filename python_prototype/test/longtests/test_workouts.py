""" Call abbreviated varsions of functions from python_prototypes/worksouts/ """

from workouts.wo_comp_mc_qmc import abstol_comparison, dimension_comparison
from workouts.wo_asian_option import test_distributions_asian_option
from workouts.wo_customizations import quick_construct_integrand
from workouts.wo_keister import test_distributions_keister

import unittest
from numpy import arange


class TestWorkouts(unittest.TestCase):

    def test_comparisons(self):
        abstol_comparison(abstols=arange(.1, .4, .1))
        dimension_comparison(dimensions=arange(1, 4))

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
