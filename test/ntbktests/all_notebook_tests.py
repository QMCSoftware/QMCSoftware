""" Unit tests for long-running demo notebooks execution using testbook """
import unittest
from testbook import testbook

class LongNotebookTests(unittest.TestCase):
    """Test class for long-running demo notebooks (> 30 seconds typical execution)"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.demos_path = '../../demos/'
    
    @testbook('../../demos/control_variates.ipynb', execute=True) 
    def test_control_variates_notebook(self, tb):
        pass
    
    @testbook('../../demos/elliptic-pde.ipynb', execute=True)
    def test_elliptic_pde_notebook(self, tb):
        pass
    
    @testbook('../../demos/nei_demo.ipynb', execute=True)
    def test_nei_demo_notebook(self, tb):
        pass

    @testbook('../../demos/qei-demo-for-blog.ipynb', execute=True)
    def test_qei_demo_blog_notebook(self, tb):
        pass
    
    @testbook('../../demos/ray_tracing.ipynb', execute=True)
    def test_ray_tracing_notebook(self, tb):
        pass

    @unittest.skip("Times out (> 600s)")
    @testbook('../../demos/asian-option-mlqmc.ipynb', execute=True)
    def test_asian_option_mlqmc_notebook(self, tb):
        pass

    @unittest.skip("API change: generalize parameter")
    @testbook('../../demos/dakota_genz.ipynb', execute=True)
    def test_dakota_genz_notebook(self, tb):
        pass

    @unittest.skip("Missing file path issue")
    @testbook('../../demos/digital_net_b2.ipynb', execute=True)
    def test_digital_net_b2_notebook(self, tb):
        pass

    @unittest.skip("Array comparison bug")
    @testbook('../../demos/gaussian_diagnostics_demo.ipynb', execute=True)
    def test_gaussian_diagnostics_demo_notebook(self, tb):
        pass

    @unittest.skip("Missing CSV file")
    @testbook('../../demos/importance_sampling.ipynb', execute=True)
    def test_importance_sampling_notebook(self, tb):
        pass

    @testbook('../../demos/integration_examples.ipynb', execute=True)
    def test_integration_examples_notebook(self, tb):
        pass

    @unittest.skip("Missing skopt dependency")
    @testbook('../../demos/iris.ipynb', execute=True)
    def test_iris_notebook(self, tb):
        pass

    @unittest.skip("Assertion error with prime numbers")
    @testbook('../../demos/lattice_random_generator.ipynb', execute=True)
    def test_lattice_random_generator_notebook(self, tb):
        pass

    @unittest.skip("Times out (> 60s) - computationally intensive")
    @testbook('../../demos/ld_randomizations_and_higher_order_nets.ipynb', execute=True)
    def test_ld_randomizations_and_higher_order_nets_notebook(self, tb):
        pass

    @testbook('../../demos/lebesgue_integration.ipynb', execute=True)
    def test_lebesgue_integration_notebook(self, tb):
        pass

    @unittest.skip("API change: generalize parameter")
    @testbook('../../demos/linear-scrambled-halton.ipynb', execute=True)
    def test_linear_scrambled_halton_notebook(self, tb):
        pass

    @unittest.skip("Missing CSV file")
    @testbook('../../demos/MC_vs_QMC.ipynb', execute=True)
    def test_mc_vs_qmc_notebook(self, tb):
        pass

    @testbook('../../demos/plot_proj_function.ipynb', execute=True)
    def test_plot_proj_function_notebook(self, tb):
        pass

    @unittest.skip("Missing LookBackOption class")
    @testbook('../../demos/PricingAsianOptions.ipynb', execute=True)
    def test_pricing_asian_options_notebook(self, tb):
        pass

    @unittest.skip("Missing matplotlib style file")
    @testbook('../../demos/prob_failure_gp_ci.ipynb', execute=True)
    def test_prob_failure_gp_ci_notebook(self, tb):
        pass

    @unittest.skip("Missing matplotlib style file")
    @testbook('../../demos/pydata.chi.2023.ipynb', execute=True)
    def test_pydata_chi_2023_notebook(self, tb):
        pass

    @testbook('../../demos/qmcpy_intro.ipynb', execute=True)
    def test_qmcpy_intro_notebook(self, tb):
        pass

    @unittest.skip("Missing CSV files")
    @testbook('../../demos/quasirandom_generators.ipynb', execute=True)
    def test_quasirandom_generators_notebook(self, tb):
        pass

    @testbook('../../demos/quickstart.ipynb', execute=True)
    def test_quickstart_notebook(self, tb):
        pass

    @unittest.skip("API change: generalize parameter")
    @testbook('../../demos/sample_scatter_plots.ipynb', execute=True)
    def test_sample_scatter_plots_notebook(self, tb):
        pass

    @testbook('../../demos/saving_qmc_state.ipynb', execute=True)
    def test_saving_qmc_state_notebook(self, tb):
        pass

    @unittest.skip("Requires external server")
    @testbook('../../demos/umbridge.ipynb', execute=True)
    def test_umbridge_notebook(self, tb):
        pass

    @unittest.skip("Times out (> 600s)")
    @testbook('../../demos/vectorized_qmc.ipynb', execute=True)
    def test_vectorized_qmc_notebook(self, tb):
        pass

    @unittest.skip("Times out (> 60s) - Bayesian inference computationally intensive")
    @testbook('../../demos/vectorized_qmc_bayes.ipynb', execute=True)
    def test_vectorized_qmc_bayes_notebook(self, tb):
        pass

    @testbook('../../demos/some_true_measures.ipynb', execute=True)
    def test_some_true_measures_notebook(self, tb):
        pass

    @unittest.skip("Missing data files")
    @testbook('../../demos/talk_paper_demos/Argonne_Talk_2023_May/Argonne_2023_Talk_Figures.ipynb', execute=True)
    def test_argonne_talk_2023_figures_notebook(self, tb):
        pass

    @unittest.skip("Missing data files")
    @testbook('../../demos/talk_paper_demos/MCQMC2022_Article_Figures/MCQMC2022_Article_Figures.ipynb', execute=True)
    def test_mcqmc2022_article_figures_notebook(self, tb):
        pass

    @unittest.skip("Missing data files")
    @testbook('../../demos/talk_paper_demos/Purdue_Talk_2023_March/Purdue_Talk_Figures.ipynb', execute=True)
    def test_purdue_talk_figures_notebook(self, tb):
        pass
    

if __name__ == '__main__':
    unittest.main()
    # python -m pytest all_notebook_tests.py 
    # Without @sunittest.skip:    Ran 33 tests in 547.652s
    # With @unittest.skip:        12 passed, 21 skipped in 144.33s (0:02:24) 

