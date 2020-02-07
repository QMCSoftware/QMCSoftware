""" Call abbreviated varsions of functions from python_prototypes/worksouts/ """

from workouts.example_constructions import barebones, distributions, integrands, measures, stopping_criteria
from workouts.integration_examples import asian_option_multi_level, asian_option_single_level, keister
from workouts.lds_sequences import python_sequences
from workouts.mc_vs_qmc import vary_abs_tol, vary_dimension
from numpy import arange
import unittest


class TestWorkouts(unittest.TestCase):

    def test_example_constructions(self):
        barebones()
        distributions()
        integrands()
        measures()
        stopping_criteria()
    
    def test_integration_examples(self):
        asian_option_multi_level()
        asian_option_single_level()
        keister()
    
    def test_lds_sequences(self):
        python_sequences()

    def test_mc_vs_qmc(self):
        vary_abs_tol()
        vary_dimension()


if __name__ == "__main__":
    unittest.main()
