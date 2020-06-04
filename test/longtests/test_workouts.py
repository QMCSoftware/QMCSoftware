""" Call abbreviated varsions of functions from python_prototypes/worksouts/ """

from workouts import *
from numpy import arange
import unittest


class TestWorkouts(unittest.TestCase):
    
    def test_integration_examples(self):
        asian_option_multi_level()
        asian_option_single_level()
        keister()
    
    def test_lds_sequences(self):
        python_sequences()

    def test_mc_vs_qmc(self):
        vary_abs_tol()
        vary_dimension()
        compare_mean_shifts()

    def test_mlmc(self):
        mcqmc06()


if __name__ == "__main__":
    unittest.main()
