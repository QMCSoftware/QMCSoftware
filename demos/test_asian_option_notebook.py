import testbook
import pytest
import numpy as np

@pytest.fixture(scope="module")
def tb():
    with testbook.testbook('asian-option-mlqmc.ipynb', execute=False) as tb:
        yield tb

def test_imports(tb):
    """Test that all imports work correctly"""
    tb.execute_cell(0)  # Import cell
    
    # Check that modules are imported
    assert 'plt' in tb.ref_globals
    assert 'np' in tb.ref_globals
    assert 'qp' in tb.ref_globals

def test_seed_setting(tb):
    """Test seed is set correctly"""
    tb.execute_cell(1)  # seed = 7
    
    seed = tb.ref("seed")
    assert seed == 7
    assert isinstance(seed, int)

def test_exact_value_computation(tb):
    """Test Asian option exact value computation"""
    tb.inject("""
    import matplotlib.pyplot as plt
    import numpy as np
    import qmcpy as qp
    seed = 7
    """)
    
    tb.execute_cell(4)  # Exact value computation cell
    
    # This cell should produce output but not create variables
    # We can check that it runs without errors
    assert True  # If we get here, cell executed successfully

def test_eval_option_function(tb):
    """Test eval_option function definition"""
    tb.inject("""
    import matplotlib.pyplot as plt
    import numpy as np
    import qmcpy as qp
    seed = 7
    """)
    
    tb.execute_cell(5)  # eval_option function definition
    
    eval_option = tb.ref("eval_option")
    assert callable(eval_option)
    
    # Test function signature
    import inspect
    sig = inspect.signature(eval_option)
    assert len(sig.parameters) == 3
    assert 'option_mc' in sig.parameters
    assert 'option_qmc' in sig.parameters
    assert 'abs_tol' in sig.parameters

def test_option_definitions(tb):
    """Test ML option definitions"""
    tb.inject("""
    import matplotlib.pyplot as plt
    import numpy as np
    import qmcpy as qp
    seed = 7
    """)
    
    tb.execute_cell(6)  # Option definitions
    
    option_mc = tb.ref("option_mc")
    option_qmc = tb.ref("option_qmc")
    
    # Check that options are created
    assert option_mc is not None
    assert option_qmc is not None
    
    # Check they have correct type
    assert hasattr(option_mc, '__class__')
    assert hasattr(option_qmc, '__class__')

def test_single_evaluation(tb):
    """Test running eval_option once"""
    tb.inject("""
    import matplotlib.pyplot as plt
    import numpy as np
    import qmcpy as qp
    seed = 7
    """)
    
    # Execute necessary cells
    tb.execute_cell(5)  # eval_option function
    tb.execute_cell(6)  # option definitions
    tb.execute_cell(7)  # Single evaluation
    
    # This should run without errors
    assert True

def test_tolerance_loop_setup(tb):
    """Test tolerance loop variables"""
    tb.inject("""
    import matplotlib.pyplot as plt
    import numpy as np
    import qmcpy as qp
    seed = 7
    """)
    
    tb.execute_cell(8)  # Tolerance loop setup
    
    repetitions = tb.ref("repetitions")
    tolerances = tb.ref("tolerances")
    levels = tb.ref("levels")
    times = tb.ref("times")
    
    # Check repetitions
    assert repetitions == 5
    assert isinstance(repetitions, int)
    
    # Check tolerances
    assert isinstance(tolerances, np.ndarray)
    assert len(tolerances) == 5
    assert np.all(tolerances > 0)
    assert tolerances[0] > tolerances[-1]  # Decreasing
    
    # Check dictionaries are initialized
    assert isinstance(levels, dict)
    assert isinstance(times, dict)

def test_average_time_computation(tb):
    """Test average time computation"""
    # First set up the necessary data
    tb.inject("""
    import numpy as np
    repetitions = 5
    tolerances = 5*np.logspace(-1, -3, num=5)
    times = {}
    # Mock some time data
    for t in range(len(tolerances)):
        for r in range(repetitions):
            times[t, r] = [np.random.rand(), np.random.rand(), 
                          np.random.rand(), np.random.rand()]
    """)
    
    tb.execute_cell(9)  # Average time computation
    
    avg_time = tb.ref("avg_time")
    
    # Check structure
    assert isinstance(avg_time, dict)
    assert len(avg_time) == 4  # 4 methods
    
    # Check each method has correct number of tolerance values
    for method in range(4):
        assert len(avg_time[method]) == 5  # 5 tolerance levels
        assert all(t >= 0 for t in avg_time[method])  # Non-negative times

def test_plotting_cells(tb):
    """Test that plotting cells execute without errors"""
    tb.inject("""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Mock data for plotting
    tolerances = 5*np.logspace(-1, -3, num=5)
    avg_time = {
        0: [1.0, 0.5, 0.1, 0.05, 0.01],
        1: [0.8, 0.4, 0.08, 0.04, 0.008],
        2: [0.9, 0.45, 0.09, 0.045, 0.009],
        3: [0.7, 0.35, 0.07, 0.035, 0.007]
    }
    """)
    
    # Execute plotting cell
    tb.execute_cell(10)
    
    # Check that plot was created
    assert len(plt.get_fignums()) > 0

def test_max_levels_computation(tb):
    """Test max levels computation"""
    tb.inject("""
    import numpy as np
    repetitions = 5
    tolerances = 5*np.logspace(-1, -3, num=5)
    levels = {}
    
    # Mock level data
    for r in range(repetitions):
        levels[len(tolerances)-1, r] = [10, 8, 9, 7]  # 4 methods
    """)
    
    tb.execute_cell(11)  # Max levels computation
    
    max_levels = tb.ref("max_levels")
    
    # Check structure
    assert isinstance(max_levels, dict)
    assert len(max_levels) == 4  # 4 methods
    
    # Check each method
    for method in range(4):
        assert len(max_levels[method]) == 15  # 15 possible levels
        assert all(0 <= frac <= 1 for frac in max_levels[method])  # Fractions
        assert abs(sum(max_levels[method]) - 1.0) < 1e-10  # Sum to 1

def test_final_plotting(tb):
    """Test final bar chart plotting"""
    tb.inject("""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Mock max_levels data
    max_levels = {
        0: [0.2, 0.3, 0.5] + [0.0]*12,
        1: [0.1, 0.4, 0.5] + [0.0]*12,
        2: [0.3, 0.3, 0.4] + [0.0]*12,
        3: [0.2, 0.2, 0.6] + [0.0]*12
    }
    """)
    
    # Execute final plotting cell
    tb.execute_cell(12)
    
    # Check that plot was created
    assert len(plt.get_fignums()) > 0

def test_notebook_execution_order(tb):
    """Test that cells can be executed in order without errors"""
    # Execute all code cells in order
    code_cells = [0, 1, 4, 5, 6, 7]  # First few cells
    
    for cell_index in code_cells:
        try:
            tb.execute_cell(cell_index)
        except Exception as e:
            pytest.fail(f"Cell {cell_index} failed to execute: {str(e)}")

def test_qmcpy_integration(tb):
    """Test that QMCPy objects are created correctly"""
    tb.inject("""
    import qmcpy as qp
    seed = 7
    """)
    
    # Test creating an Asian option
    tb.inject("""
    test_option = qp.AsianOption(
        qp.Sobol(4, seed=seed), 
        volatility=.2, 
        start_price=100, 
        strike_price=100, 
        interest_rate=.05
    )
    """)
    
    test_option = tb.ref("test_option")
    assert test_option is not None
    assert hasattr(test_option, 'volatility')

def test_numerical_stability(tb):
    """Test that numerical computations are stable"""
    tb.inject("""
    import numpy as np
    
    # Test tolerance array
    tolerances = 5*np.logspace(-1, -3, num=5)
    """)
    
    tolerances = tb.ref("tolerances")
    
    # Check no NaN or inf values
    assert not np.any(np.isnan(tolerances))
    assert not np.any(np.isinf(tolerances))
    
    # Check reasonable range
    assert np.all(tolerances > 0)
    assert np.all(tolerances < 10)

def test_data_structures(tb):
    """Test that data structures are properly initialized"""
    tb.inject("""
    levels = {}
    times = {}
    
    # Add some test data
    levels[0, 0] = 5
    times[0, 0] = [1.2, 0.8, 0.5, 0.3]
    """)
    
    levels = tb.ref("levels")
    times = tb.ref("times")
    
    # Check dictionary access
    assert (0, 0) in levels
    assert (0, 0) in times
    assert levels[0, 0] == 5
    assert len(times[0, 0]) == 4

# Optional: Test for specific QMCPy algorithm behavior
def test_algorithm_comparison(tb):
    """Test that different algorithms produce different results"""
    tb.inject("""
    import qmcpy as qp
    import numpy as np
    
    # Create simple test case
    option = qp.MLCallOptions(qp.IIDStdUniform(seed=7), option="asian")
    abs_tol = 0.01
    
    # Run different algorithms
    results = {}
    algs = {
        "MLMC": qp.CubMCML(option, abs_tol=abs_tol, levels_max=5),
        "MLQMC": qp.CubQMCML(option, abs_tol=abs_tol, levels_max=5)
    }
    
    for name, alg in algs.items():
        sol, data = alg.integrate()
        results[name] = {"solution": sol, "levels": data.levels}
    """)
    
    results = tb.ref("results")
    
    # Check that results differ (they should use different methods)
    assert "MLMC" in results
    assert "MLQMC" in results
    
    # Solutions should be close but potentially different
    mlmc_sol = results["MLMC"]["solution"]
    mlqmc_sol = results["MLQMC"]["solution"]
    
    assert abs(mlmc_sol - mlqmc_sol) < 0.1  # Should be close
    assert isinstance(results["MLMC"]["levels"], int)
    assert isinstance(results["MLQMC"]["levels"], int)
