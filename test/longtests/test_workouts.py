""" Call abbreviated varsions of functions from python_prototypes/worksouts/ """

from workouts import *
import sys
vinvo = sys.version_info
if vinvo[0]==3: import unittest
else: import unittest2 as unittest

class TestWorkouts(unittest.TestCase):
    
    def test_integration_examples(self):
        asian_option_multi_level()
        asian_option_single_level()
        keister()
        pi_problem()
    
    def test_lds_sequences(self):
        python_sequences()

    def test_mc_vs_qmc(self):
        vary_abs_tol()
        vary_dimension()
        compare_mean_shifts()

    def test_mlmc(self):
        mcqmc06()
        european_options()


if __name__ == "__main__":
    unittest.main()
