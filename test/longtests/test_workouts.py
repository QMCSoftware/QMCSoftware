from workouts import *
import unittest


class TestWorkouts(unittest.TestCase):
    
    def test_integration_examples(self):
        asian_option_multi_level()
        asian_option_single_level()
        asian_option_single_level_high_dimensions()
        keister()
        pi_problem()
        pi_problem_bayes_net()
    
    def test_lds_sequences(self):
        python_sequences()

    def test_mc_vs_qmc(self):
        import warnings
        warnings.simplefilter('ignore',RuntimeWarning)
        vary_abs_tol()
        vary_dimension()
        compare_mean_shifts()

    def test_mlmc(self):
        mcqmc06()
        european_options()


if __name__ == "__main__":
    unittest.main()
