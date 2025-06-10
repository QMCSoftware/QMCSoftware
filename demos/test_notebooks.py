import testbook
import pytest
import numpy as np
import pandas as pd
import inspect

# ============================================
# Tests for asian-option-mlqmc.ipynb
# ============================================

class TestAsianOptionMLQMC:
	@pytest.fixture(scope="class")
	def tb(self):
		with testbook.testbook('asian-option-mlqmc.ipynb', execute=False) as tb:
			yield tb

	def test_imports(self, tb):
		"""Test that all imports work correctly"""
		tb.execute_cell(0)
		assert 'plt' in tb.ref_globals
		assert 'np' in tb.ref_globals
		assert 'qp' in tb.ref_globals

	def test_seed_setting(self, tb):
		"""Test seed is set correctly"""
		tb.execute_cell(1)
		seed = tb.ref("seed")
		assert seed == 7
		assert isinstance(seed, int)

	def test_exact_value_computation(self, tb):
		"""Test Asian option exact value computation"""
		tb.inject("import matplotlib.pyplot as plt; import numpy as np; import qmcpy as qp; seed = 7")
		tb.execute_cell(4)
		assert True  # If we get here, cell executed successfully

	def test_eval_option_function(self, tb):
		"""Test eval_option function definition"""
		tb.inject("import matplotlib.pyplot as plt; import numpy as np; import qmcpy as qp; seed = 7")
		tb.execute_cell(5)
		eval_option = tb.ref("eval_option")
		assert callable(eval_option)
		sig = inspect.signature(eval_option)
		assert len(sig.parameters) == 3

	def test_option_definitions(self, tb):
		"""Test ML option definitions"""
		tb.inject("import matplotlib.pyplot as plt; import numpy as np; import qmcpy as qp; seed = 7")
		tb.execute_cell(6)
		option_mc = tb.ref("option_mc")
		option_qmc = tb.ref("option_qmc")
		assert option_mc is not None
		assert option_qmc is not None

# ============================================
# Tests for control_variates.ipynb
# ============================================

class TestControlVariates:
	@pytest.fixture(scope="class")
	def tb(self):
		with testbook.testbook('control_variates.ipynb', execute=False) as tb:
			yield tb

	def test_imports(self, tb):
		"""Test that all imports work correctly"""
		tb.execute_cell(0)
		assert 'qmcpy' in tb.ref_globals
		assert 'numpy' in tb.ref_globals

	def test_matplotlib_setup(self, tb):
		"""Test matplotlib configuration"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		pyplot = tb.ref("pyplot")
		assert pyplot is not None
		tb.inject("assert pyplot.rcParams['font.size'] == 20")

	def test_compare_function(self, tb):
		"""Test compare function definition"""
		tb.execute_cell(0)
		tb.execute_cell(2)
		compare = tb.ref("compare")
		assert callable(compare)
		sig = inspect.signature(compare)
		assert len(sig.parameters) == 4

	def test_polynomial_problem(self, tb):
		"""Test polynomial problem setup and execution"""
		tb.inject("from qmcpy import *; from numpy import *")
		tb.execute_cell(2)
		tb.execute_cell(3)
		poly_problem = tb.ref("poly_problem")
		assert callable(poly_problem)
		tb.inject("""
		test_distrib = IIDStdUniform(3, seed=7)
		g1, cvs, cvmus = poly_problem(test_distrib)
		assert len(cvs) == 2
		assert len(cvmus) == 2
		assert cvmus == [1, 4/3]
		""")

	def test_keister_problem(self, tb):
		"""Test Keister problem setup"""
		tb.inject("from qmcpy import *; from numpy import *")
		tb.execute_cell(2)
		tb.execute_cell(4)
		keister_problem = tb.ref("keister_problem")
		assert callable(keister_problem)

	def test_option_problem(self, tb):
		"""Test option pricing problem setup"""
		tb.inject("from qmcpy import *; from numpy import *")
		tb.execute_cell(2)
		tb.execute_cell(5)
		call_put = tb.ref("call_put")
		assert call_put == 'call'
		start_price = tb.ref("start_price")
		assert start_price == 100

# ============================================
# Tests for dakota_genz.ipynb
# ============================================

class TestDakotaGenz:
	@pytest.fixture(scope="class")
	def tb(self):
		with testbook.testbook('dakota_genz.ipynb', execute=False) as tb:
			yield tb

	def test_imports_and_setup(self, tb):
		"""Test imports and matplotlib style setup"""
		tb.execute_cell(0)
		assert 'numpy' in tb.ref_globals
		assert 'qmcpy' in tb.ref_globals
		assert 'pd' in tb.ref_globals

	def test_dimension_and_sample_arrays(self, tb):
		"""Test dimension and sample size arrays"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		kinds_func = tb.ref("kinds_func")
		kinds_coeff = tb.ref("kinds_coeff")
		ds = tb.ref("ds")
		ns = tb.ref("ns")
		assert kinds_func == ['oscillatory', 'corner-peak']
		assert kinds_coeff == [1, 2, 3]
		assert len(ds) == 8
		assert ds[0] == 1
		assert ds[-1] == 128

	def test_reference_solutions_computation(self, tb):
		"""Test reference solutions computation"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		tb.inject("ds = 2**np.arange(3); x_full = DigitalNetB2(ds.max(), seed=7).gen_samples(2**10)")
		tb.execute_cell(2)
		ref_sols = tb.ref("ref_sols")
		assert isinstance(ref_sols, pd.DataFrame)
		assert 'd' == ref_sols.index.name

	def test_dakota_file_handling(self, tb):
		"""Test Dakota file loading (expected to fail without file)"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		tb.execute_cell(2)
		with pytest.raises(Exception) as exc_info:
			tb.execute_cell(3)
		assert "FileNotFoundError" in str(exc_info.typename)

# ============================================
# Tests for integration_examples.ipynb
# ============================================

class TestIntegrationExamples:
	@pytest.fixture(scope="class")
	def tb(self):
		with testbook.testbook('integration_examples.ipynb', execute=False) as tb:
			yield tb

	def test_imports(self, tb):
		"""Test that imports work correctly"""
		tb.execute_cell(0)
		assert 'qmcpy' in tb.ref_globals
		assert 'arange' in tb.ref_globals

	def test_keister_example(self, tb):
		"""Test Keister example execution"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		tb.inject("""
		assert 'data' in globals()
		assert 'solution' in globals()
		assert isinstance(solution, (int, float))
		assert 0 < solution < 10
		""")

	def test_asian_option_single_level(self, tb):
		"""Test Asian option single level"""
		tb.execute_cell(0)
		tb.execute_cell(2)
		tb.inject("""
		assert isinstance(solution, (int, float))
		assert solution > 0
		assert hasattr(data, 'time_integrate')
		""")

	def test_asian_option_multi_level(self, tb):
		"""Test Asian option multi-level"""
		tb.execute_cell(0)
		tb.execute_cell(3)
		tb.inject("""
		assert hasattr(data, 'levels')
		assert data.levels > 1
		assert np.array_equal(data.integrand.multilevel_dims, [4, 8, 16])
		""")

	def test_bayesian_cubature_lattice(self, tb):
		"""Test Bayesian cubature with lattice"""
		tb.execute_cell(0)
		tb.execute_cell(4)
		tb.inject("""
		assert 'solution' in globals()
		assert data.comb_bound_low <= solution <= data.comb_bound_high
		""")

	def test_bayesian_cubature_sobol(self, tb):
		"""Test Bayesian cubature with Sobol"""
		tb.execute_cell(0)
		tb.execute_cell(5)
		tb.inject("""
		assert isinstance(data.integrand.discrete_distrib, Sobol)
		assert data.integrand.discrete_distrib.graycode == False
		""")

# ============================================
# Tests for lebesgue_integration.ipynb
# ============================================

class TestLebesgueIntegration:
	@pytest.fixture(scope="class")
	def tb(self):
		with testbook.testbook('lebesgue_integration.ipynb', execute=False) as tb:
			yield tb

	def test_imports(self, tb):
		"""Test imports"""
		tb.execute_cell(0)
		assert 'qmcpy' in tb.ref_globals
		assert 'numpy' in tb.ref_globals

	def test_sample_problem_1_setup(self, tb):
		"""Test sample problem 1 parameters"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		abs_tol = tb.ref("abs_tol")
		dim = tb.ref("dim")
		true_value = tb.ref("true_value")
		assert abs_tol == 0.01
		assert dim == 1
		assert abs(true_value - 8/3) < 1e-10

	def test_lebesgue_measure_1d(self, tb):
		"""Test Lebesgue measure integration"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		tb.execute_cell(2)
		tb.inject("assert abs(solution - true_value) <= abs_tol")

	def test_uniform_measure_1d(self, tb):
		"""Test uniform measure integration"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		tb.execute_cell(3)
		tb.inject("assert abs(solution - true_value) <= abs_tol")

	def test_sample_problem_2_setup(self, tb):
		"""Test sample problem 2 parameters"""
		tb.execute_cell(0)
		tb.execute_cell(4)
		tb.inject("""
		assert dim == 2
		assert len(a) == 2
		assert len(b) == 2
		assert abs(true_value - 23.33333) < 0.00001
		""")

	def test_sample_problem_3(self, tb):
		"""Test sin(x)/log(x) integral"""
		tb.execute_cell(0)
		tb.execute_cell(7)
		tb.execute_cell(8)
		tb.inject("""
		assert abs_tol == 0.0001
		assert abs(solution - true_value) <= abs_tol
		""")

	def test_sample_problem_4(self, tb):
		"""Test integral over R^d"""
		tb.execute_cell(0)
		tb.execute_cell(9)
		tb.execute_cell(10)
		tb.inject("""
		assert abs_tol == 0.1
		assert abs(solution - true_value) <= abs_tol
		""")

# ============================================
# Tests for PricingAsianOptions.ipynb
# ============================================

class TestPricingAsianOptions:
	@pytest.fixture(scope="class")
	def tb(self):
		with testbook.testbook('PricingAsianOptions.ipynb', execute=False) as tb:
			yield tb

	def test_imports(self, tb):
		"""Test imports"""
		tb.execute_cell(0)
		assert 'qp' in tb.ref_globals
		assert 'np' in tb.ref_globals
		assert 'stats' in tb.ref_globals

	def test_european_option_parameters(self, tb):
		"""Test European option parameters"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		params = {
			'initPrice': 120,
			'interest': 0.02,
			'vol': 0.5,
			'callput': 'call',
			'strike': 130,
			'tfinal': 0.25,
			'd': 12,
			'absTol': 0.05,
			'relTol': 0,
			'sampleSize': 10**6
		}
		for name, expected in params.items():
			actual = tb.ref(name)
			assert actual == expected, f"{name} should be {expected}"

	def test_european_option_pricing(self, tb):
		"""Test European option pricing"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		tb.execute_cell(2)
		tb.inject("""
		assert hasattr(EuroCall, 'get_exact_value')
		exact_val = EuroCall.get_exact_value()
		assert isinstance(exact_val, (int, float))
		assert exact_val > 0
		""")

	def test_arithmetic_mean_call(self, tb):
		"""Test arithmetic mean call option"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		tb.execute_cell(3)
		tb.inject("""
		assert ArithMeanCall.mean_type == 'arithmetic'
		assert ArithMeanCall.call_put == 'call'
		""")

	def test_barrier_option(self, tb):
		"""Test barrier option"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		tb.execute_cell(8)
		tb.inject("""
		assert barrier == 150
		assert BarrierUpInCall.barrier_price == 150
		assert BarrierUpInCall.in_out == 'in'
		""")

# ============================================
# Tests for qmcpy_intro.ipynb
# ============================================

class TestQMCPyIntro:
	@pytest.fixture(scope="class")
	def tb(self):
		with testbook.testbook('qmcpy_intro.ipynb', execute=False) as tb:
			yield tb

	def test_import_methods(self, tb):
		"""Test different import methods"""
		tb.execute_cell(0)
		tb.inject("""
		assert 'qp' in globals()
		assert hasattr(qp, 'name')
		assert qp.name == 'qmcpy'
		""")

	def test_individual_imports(self, tb):
		"""Test importing individual modules"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		tb.inject("""
		test_classes = ['Keister', 'AsianOption', 'Uniform', 'Gaussian']
		for cls_name in test_classes:
			assert cls_name in globals() or hasattr(qmcpy, cls_name)
		""")

	def test_lattice_generation(self, tb):
		"""Test lattice sequence generation"""
		for i in range(4):
			tb.execute_cell(i)
		tb.inject("""
		samples = distribution.gen_samples(n_min=0, n_max=4)
		assert samples.shape == (4, 2)
		assert np.all((samples >= 0) & (samples <= 1))
		""")

	def test_function_definition_attempts(self, tb):
		"""Test the function definition process"""
		for i in range(7):
			tb.execute_cell(i)
		tb.inject("""
		result = f(0.01)
		assert isinstance(result, (int, float))
		assert result > 0
		""")

	def test_corrected_function(self, tb):
		"""Test the corrected function"""
		for i in range(13):
			tb.execute_cell(i)
		tb.inject("""
		x_test = array([[1., 0.], [0., 0.01], [0.04, 0.04]])
		result = myfunc(x_test)
		assert len(result) == 3
		assert np.all(np.isfinite(result))
		""")

	def test_integration_1d(self, tb):
		"""Test 1D integration"""
		for i in range(14):
			tb.execute_cell(i)
		tb.inject("""
		assert 'solution' in globals()
		assert 0.6 < solution < 0.7
		""")

	def test_integration_2d(self, tb):
		"""Test 2D integration"""
		for i in range(16):
			tb.execute_cell(i)
		tb.inject("""
		assert dim == 2
		assert 'solution2' in globals()
		assert 0.8 < solution2 < 0.85
		""")

# ============================================
# Tests for quickstart.ipynb
# ============================================

class TestQuickstart:
	@pytest.fixture(scope="class")
	def tb(self):
		with testbook.testbook('quickstart.ipynb', execute=False) as tb:
			yield tb

	def test_keister_function_definition(self, tb):
		"""Test Keister function implementation"""
		tb.execute_cell(0)
		tb.inject("""
		x = np.array([[1.0, 0.0]])
		k = keister(x)
		assert k.shape == (1,)
		assert k[0] == np.pi

		x = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
		k = keister(x)
		assert k.shape == (3,)
		""")

	def test_qmcpy_setup(self, tb):
		"""Test QMCPy setup for Keister example"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		tb.inject("""
		assert d == 2
		assert isinstance(discrete_distrib, qmcpy.Lattice)
		assert discrete_distrib.d == 2
		assert isinstance(true_measure, qmcpy.Gaussian)
		assert true_measure.mean == 0
		assert true_measure.covariance == 0.5
		assert isinstance(integrand, qmcpy.CustomFun)
		assert isinstance(stopping_criterion, qmcpy.CubQMCLatticeG)
		assert stopping_criterion.abs_tol == 1e-3
		""")

	def test_integration(self, tb):
		"""Test integration execution"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		tb.execute_cell(2)
		tb.inject("""
		assert 'solution' in globals()
		assert 'data' in globals()
		assert 1.7 < solution < 1.9
		assert hasattr(data, 'n_total')
		assert data.n_total > 0
		assert data.comb_bound_low <= solution <= data.comb_bound_high
		""")

# ============================================
# Tests for saving_qmc_state.ipynb
# ============================================

class TestSavingQMCState:
	@pytest.fixture(scope="class")
	def tb(self):
		with testbook.testbook('saving_qmc_state.ipynb', execute=False) as tb:
			yield tb

	def test_imports_and_setup(self, tb):
		"""Test imports and matplotlib setup"""
		tb.execute_cell(0)
		tb.inject("""
		assert 'qmcpy' in globals()
		assert 'np' in globals()
		assert 'pyplot' in globals()
		assert 'norm' in globals()
		assert lw == 3
		""")

	def test_iid_keister_integration(self, tb):
		"""Test IID Keister integration with CubMCCLT"""
		tb.execute_cell(0)
		tb.execute_cell(1)
		tb.inject("""
		assert abs_tol == 0.3
		assert isinstance(integrand, Keister)
		assert integrand.d == 2
		assert isinstance(integrand.discrete_distrib, IIDStdUniform)
		assert 'solution' in globals()
		assert 0 < solution < 3
		assert data.n_total > 0
		""")

	def test_lattice_keister_setup(self, tb):
		"""Test lattice Keister setup and Bayesian integration"""
		tb.execute_cell(0)
		tb.execute_cell(3)
		tb.inject("""
		assert isinstance(k, Keister)
		assert isinstance(k.discrete_distrib, Lattice)
		assert k.discrete_distrib.d == 2
		assert k.discrete_distrib.order == 'linear'
		assert k.discrete_distrib.seed == 123456789
		assert isinstance(sc, CubBayesLatticeG)
		assert sc.abs_tol == 0.05
		assert 1.5 < solution < 2.0
		""")

	def test_object_introspection(self, tb):
		"""Test object introspection with dir()"""
		tb.execute_cell(0)
		tb.execute_cell(3)
		tb.execute_cell(4)
		tb.inject("""
		k_attrs = dir(k)
		essential_attrs = ['discrete_distrib', 'true_measure', 'f', 'g', 'd', 'parameters']
		for attr in essential_attrs:
			assert attr in k_attrs
		""")

# ============================================
# Cross-notebook integration tests
# ============================================

class TestCrossNotebookIntegration:
	def test_keister_consistency_across_notebooks(self):
		"""Test that Keister function gives consistent results across notebooks"""
		results = []
		tolerances = []

		# From quickstart.ipynb
		with testbook.testbook('quickstart.ipynb', execute=False) as tb:
			for i in range(3):
				tb.execute_cell(i)
			tb.inject("quickstart_sol = solution; quickstart_tol = stopping_criterion.abs_tol")
			results.append(tb.ref("quickstart_sol"))
			tolerances.append(tb.ref("quickstart_tol"))

		# From integration_examples.ipynb
		with testbook.testbook('integration_examples.ipynb', execute=False) as tb:
			tb.execute_cell(0)
			tb.execute_cell(1)
			tb.inject("examples_sol = solution; examples_tol = data.stopping_crit.abs_tol")
			results.append(tb.ref("examples_sol"))
			tolerances.append(tb.ref("examples_tol"))

		# All results should be reasonable for Keister function
		assert all(1.5 < r < 2.5 for r in results)
		assert all(t > 0 for t in tolerances)

	def test_asian_option_consistency(self):
		"""Test Asian option consistency across notebooks"""
		# From integration_examples.ipynb
		with testbook.testbook('integration_examples.ipynb', execute=False) as tb:
			tb.execute_cell(0)
			tb.execute_cell(2)  # Single level Asian
			tb.inject("single_level_sol = solution")
			single_level = tb.ref("single_level_sol")

		# Check reasonable bounds for option price
		assert 0 < single_level < 100

# ============================================
# Parametrized tests
# ============================================

@pytest.mark.parametrize("notebook,min_cells", [
	("asian-option-mlqmc.ipynb", 10),
	("control_variates.ipynb", 5),
	("dakota_genz.ipynb", 4),
	("integration_examples.ipynb", 5),
	("lebesgue_integration.ipynb", 10),
	("PricingAsianOptions.ipynb", 8),
	("qmcpy_intro.ipynb", 15),
	("quickstart.ipynb", 3),
	("saving_qmc_state.ipynb", 5),
])
def test_notebook_has_minimum_cells(notebook, min_cells):
	"""Test that notebooks have minimum expected cells"""
	with testbook.testbook(notebook, execute=False) as tb:
		assert len(tb.cells) >= min_cells

# ============================================
# Utility tests
# ============================================

class TestUtilities:
	def test_numpy_functionality(self):
		"""Test numpy is available and functional"""
		import numpy as np
		x = np.array([1, 2, 3])
		assert x.sum() == 6
		assert np.sqrt(4) == 2

	def test_qmcpy_imports(self):
		"""Test all major QMCPy classes can be imported"""
		from qmcpy import (
			Keister, AsianOption, EuropeanOption,
			Uniform, Gaussian, Lebesgue,
			IIDStdUniform, Lattice, Sobol, Halton,
			CubMCCLT, CubQMCSobolG, CubBayesLatticeG
		)
		assert all([Keister, AsianOption, Uniform, IIDStdUniform, CubMCCLT])

	def test_pandas_functionality(self):
		"""Test pandas is available"""
		import pandas as pd
		df = pd.DataFrame({'a': [1, 2, 3]})
		assert len(df) == 3

# ============================================
# Performance and stability tests
# ============================================

class TestPerformanceStability:
	def test_small_integration_problems(self):
		"""Test that small integration problems complete quickly"""
		import time
		from qmcpy import Keister, IIDStdUniform, CubMCCLT

		start = time.time()
		k = Keister(IIDStdUniform(2, seed=7))
		sol, data = CubMCCLT(k, abs_tol=0.5).integrate()
		duration = time.time() - start

		assert duration < 5.0  # Should complete in less than 5 seconds
		assert isinstance(sol, (int, float))
		assert data.n_total < 1e6  # Shouldn't need too many samples for low accuracy

import testbook
import numpy as np
import pandas as pd
import matplotlib
import pytest

# #############################################################################
# ## Tests for quasirandom_generators.ipynb
# #############################################################################

class TestQuasirandomGenerators:
	@pytest.fixture(scope="class")
	def tb(self):
		"""Fixture to execute the notebook for this class."""
		with testbook.testbook("quasirandom_generators.ipynb", execute=True) as tb_obj:
			yield tb_obj

	def test_cell_1_markdown(self, tb):
		"""Test for markdown cell 1: Title"""
		assert tb.nb.cells[0].cell_type == "markdown"

	def test_cell_2_imports(self, tb):
		"""Test for code cell 2: Imports and setup"""
		tb.inject(
			"""
			assert 'Lattice' in locals()
			assert 'pd' in locals()
			assert 'plt' in locals()
			assert plt.rcParams['font.size'] == 14
			""",
			cell_index=1
		)

	def test_cell_4_iid_generator(self, tb):
		"""Test for code cell 4: IID generator and plot"""
		d = tb.ref("d")
		n = tb.ref("n")
		s = tb.ref("s")
		assert d == 2 and n == 2**s
		iid_generator = tb.ref("iid_generator")
		assert iid_generator.__class__.__name__ == 'IID'
		iid_samples = tb.ref("iid_samples")
		assert isinstance(iid_samples, np.ndarray)
		assert iid_samples.shape == (n, d)
		fig = tb.ref("fig")
		assert isinstance(fig, matplotlib.figure.Figure)
		ax = tb.ref("ax")
		assert isinstance(ax, matplotlib.axes.Axes)
		assert "IID" in ax.get_title()

	def test_cell_6_lattice_generator(self, tb):
		"""Test for code cell 6: Lattice generator and plot"""
		lattice_generator = tb.ref("lattice_generator")
		assert lattice_generator.__class__.__name__ == 'Lattice'
		lattice_samples = tb.ref("lattice_samples")
		assert isinstance(lattice_samples, np.ndarray)
		ax = tb.ref("ax")
		assert "Lattice" in ax.get_title()

	def test_cell_8_sobol_generator(self, tb):
		"""Test for code cell 8: Sobol generator and plot"""
		sobol_generator = tb.ref("sobol_generator")
		assert sobol_generator.__class__.__name__ == 'Sobol'
		sobol_samples = tb.ref("sobol_samples")
		assert isinstance(sobol_samples, np.ndarray)
		ax = tb.ref("ax")
		assert "Sobol" in ax.get_title()


# #############################################################################
# ## Tests for some_true_measures.ipynb
# #############################################################################

class TestSomeTrueMeasures:
	@pytest.fixture(scope="class")
	def tb(self):
		"""Fixture to execute the notebook for this class."""
		with testbook.testbook("some_true_measures.ipynb", execute=True) as tb_obj:
			yield tb_obj

	def test_cell_3_imports(self, tb):
		"""Test for code cell 3: Imports"""
		tb.inject(
			"""
			assert 'CubMCG' in locals()
			assert 'Keister' in locals()
			assert 'pi' in locals()
			""",
			cell_index=2
		)

	def test_cell_4_keister_std(self, tb):
		"""Test for code cell 4: Standard Keister integration"""
		keister_std = tb.ref("keister_std")
		assert keister_std.__class__.__name__ == 'Keister'
		sol = tb.ref("sol")
		assert isinstance(sol, float)
		data = tb.ref("data")
		assert data.__class__.__name__ == 'StoppingConditionData'

	def test_cell_11_plotting(self, tb):
		"""Test for code cell 11: Plotting Keister functions"""
		fig = tb.ref("fig")
		ax = tb.ref("ax")
		assert isinstance(fig, matplotlib.figure.Figure)
		assert isinstance(ax, matplotlib.axes.Axes)
		assert ax.get_title() == 'functions'
		assert len(ax.get_lines()) == 4 # Check that 4 functions were plotted


# #############################################################################
# ## Tests for nei_demo.ipynb
# #############################################################################

class TestNeiDemo:
	@pytest.fixture(scope="class")
	def tb(self):
		"""Fixture to execute the notebook for this class."""
		with testbook.testbook("nei_demo.ipynb", execute=True) as tb_obj:
			yield tb_obj

	def test_cell_2_imports(self, tb):
		"""Test for code cell 2: Imports and setup"""
		tb.inject(
			"""
			assert 'norm' in locals()
			assert lw == 3
			assert ms == 8
			""",
			cell_index=1
		)

	def test_cell_4_fake_data(self, tb):
		"""Test for code cell 4: Fake data generation"""
		xt = tb.ref("xt")
		yt = tb.ref("yt")
		y_max = tb.ref("y_max")
		assert isinstance(xt, np.ndarray) and xt.ndim == 2
		assert isinstance(yt, np.ndarray) and yt.ndim == 1
		assert isinstance(y_max, float)

	def test_cell_6_covariance_function(self, tb):
		"""Test for code cell 6: Covariance function definition"""
		k_func = tb.ref("k")
		assert callable(k_func)
		# Test function with sample data
		x1 = np.array([[0.5]])
		x2 = np.array([[0.6]])
		l = 0.4
		expected_output = np.exp(-.5 * (np.sum((x1 - x2)**2)) / l**2)
		assert np.isclose(k_func(x1, x2, l), expected_output)

	def test_cell_13_calculation_loop(self, tb):
		"""Test for code cell 13: NEI calculation loop"""
		vals = tb.ref("vals")
		assert isinstance(vals, dict)
		assert "qmc" in vals
		assert "mc" in vals
		assert isinstance(vals["qmc"], np.ndarray)

	def test_cell_14_plotting(self, tb):
		"""Test for code cell 14: Plotting results"""
		output_text = tb.nb.cells[13].outputs[0]['data']['text/plain']
		assert "<Figure size" in output_text


# #############################################################################
# ## Tests for pydata.chi.2023.ipynb
# #############################################################################

class TestPydataChi2023:
	@pytest.fixture(scope="class")
	def tb(self):
		"""Fixture to execute the notebook for this class."""
		with testbook.testbook("pydata.chi.2023.ipynb", execute=True) as tb_obj:
			yield tb_obj

	def test_cell_3_sobol(self, tb):
		"""Test for code cell 3: Sobol sequence generation"""
		s = tb.ref("s")
		assert isinstance(s, np.ndarray)
		assert s.shape == (2**8, 2)
		ax = tb.ref("ax")
		assert "Sobol" in ax.get_title()

	def test_cell_8_asian_option(self, tb):
		"""Test for code cell 8: Asian Option Integration"""
		integrand = tb.ref("integrand")
		assert integrand.__class__.__name__ == 'AsianOption'
		solution = tb.ref("solution")
		assert isinstance(solution, np.ndarray)
		assert len(solution) > 1

	@pytest.mark.skip(reason="Requires penguins.csv file.")
	def test_cell_11_load_data(self, tb):
		"""Test for code cell 11: Loading penguin data"""
		df = tb.ref("df")
		assert isinstance(df, pd.DataFrame)
		X = tb.ref("X")
		y = tb.ref("y")
		assert isinstance(X, np.ndarray)
		assert isinstance(y, np.ndarray)

	@pytest.mark.skip(reason="Requires penguins.csv file.")
	def test_cell_13_nnkernel(self, tb):
		"""Test for code cell 13: QMC integration with NNKernel"""
		data = tb.ref("nn_sis_data")
		assert data.__class__.__name__ == 'StoppingConditionData'
		assert "NNKernel" in str(data.integrand)


# #############################################################################
# ## Tests for sample_scatter_plots.ipynb
# #############################################################################

class TestSampleScatterPlots:
	@pytest.fixture(scope="class")
	def tb(self):
		"""Fixture to execute the notebook for this class."""
		with testbook.testbook("sample_scatter_plots.ipynb", execute=True) as tb_obj:
			yield tb_obj

	def test_cell_3_make_level_plot(self, tb):
		"""Test for code cell 3: make_level_plot function definition"""
		assert callable(tb.ref("make_level_plot"))

	def test_cell_4_make_scatter_plot(self, tb):
		"""Test for code cell 4: make_scatter_plot function definition"""
		assert callable(tb.ref("make_scatter_plot"))

	def test_cell_6_european_option(self, tb):
		"""Test for code cell 6: European Option plotting loop"""
		# The cell creates multiple plots in a loop, we check the final state
		integrand = tb.ref("integrand")
		assert integrand.__class__.__name__ == 'EuropeanOption'
		fig = tb.ref("fig")
		assert isinstance(fig, matplotlib.figure.Figure)
		assert len(fig.axes) == 4 # 4 subplots

	def test_cell_8_asian_option(self, tb):
		"""Test for code cell 8: Asian Option plotting loop"""
		integrand = tb.ref("integrand")
		assert integrand.__class__.__name__ == 'AsianOption'
		fig = tb.ref("fig")
		assert isinstance(fig, matplotlib.figure.Figure)
		assert len(fig.axes) == 4


# #############################################################################
# ## Tests for linear-scrambled-halton.ipynb
# #############################################################################

class TestLinearScrambledHalton:
	@pytest.fixture(scope="class")
	def tb(self):
		"""Fixture to execute the notebook for this class."""
		with testbook.testbook("linear-scrambled-halton.ipynb", execute=True) as tb_obj:
			yield tb_obj

	def test_cell_11_qrng(self, tb):
		"""Test for code cell 11: QRNG Halton"""
		halton_qrng = tb.ref("halton_qrng")
		assert halton_qrng.__class__.__name__ == 'Halton'
		assert halton_qrng.randomize == 'QRNG'

	def test_cell_20_lms(self, tb):
		"""Test for code cell 20: LMS Scrambled Halton"""
		halton_lms = tb.ref("halton_lms")
		assert halton_lms.randomize == 'LMS'

	def test_cell_26_ds(self, tb):
		"""Test for code cell 26: Digital Shift Halton"""
		halton_ds = tb.ref("halton_ds")
		assert halton_ds.randomize == 'DS'

	def test_cell_30_lms_ds(self, tb):
		"""Test for code cell 30: LMS + DS Halton"""
		halton_lms_ds = tb.ref("halton_lms_ds")
		assert halton_lms_ds.randomize == 'LMS_DS'

	def test_cell_35_owen(self, tb):
		"""Test for code cell 35: Owen Scrambled Halton"""
		halton_owen = tb.ref("halton_owen")
		assert halton_owen.randomize == 'OWEN'

	def test_cell_46_time_comparison(self, tb):
		"""Test for code cell 46: Time comparison loop"""
		output = tb.nb.cells[45].outputs[0].text
		assert "Time to generate samples for QRNG" in output
		assert "Time to generate samples for OWEN" in output
		assert "Time to generate samples for LMS" in output
		assert "Time to generate samples for DS" in output
		assert "Time to generate samples for LMS_DS" in output

if __name__ == "__main__":
	# Run all tests
	pytest.main([__file__, "-v"])
