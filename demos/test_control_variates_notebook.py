# test_control_variates_notebook.py
import testbook
import pytest
import numpy as np

@pytest.fixture(scope="module")
def tb():
    with testbook.testbook('control_variates.ipynb', execute=False) as tb:
        yield tb

def test_imports(tb):
    """Test that all imports work correctly"""
    tb.execute_cell(0)  # Import cell
    
    # Check that modules are imported
    assert 'qmcpy' in tb.ref_globals
    assert 'numpy' in tb.ref_globals
    
    # Check specific imports
    tb.inject("assert 'array' in dir()")
    tb.inject("assert 'CustomFun' in dir(qmcpy)")

def test_matplotlib_setup(tb):
    """Test matplotlib configuration"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(1)  # matplotlib setup
    
    pyplot = tb.ref("pyplot")
    assert pyplot is not None
    
    # Check that rcParams were updated
    tb.inject("assert pyplot.rcParams['font.size'] == 20")

def test_compare_function(tb):
    """Test compare function definition"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(2)  # compare function
    
    compare = tb.ref("compare")
    assert callable(compare)
    
    # Check function signature
    import inspect
    sig = inspect.signature(compare)
    assert len(sig.parameters) == 4
    assert list(sig.parameters.keys()) == ['problem', 'discrete_distrib', 'stopping_crit', 'abs_tol']

def test_polynomial_problem(tb):
    """Test polynomial problem setup and execution"""
    tb.inject("""
    from qmcpy import *
    from numpy import *
    """)
    
    # Execute necessary cells
    tb.execute_cell(2)  # compare function
    tb.execute_cell(3)  # polynomial problem
    
    # Check that poly_problem function exists
    poly_problem = tb.ref("poly_problem")
    assert callable(poly_problem)
    
    # Test the function returns correct structure
    tb.inject("""
    test_distrib = IIDStdUniform(3, seed=7)
    g1, cvs, cvmus = poly_problem(test_distrib)
    
    assert g1 is not None
    assert len(cvs) == 2  # Two control variates
    assert len(cvmus) == 2  # Two means
    assert cvmus == [1, 4/3]  # Expected means
    """)

def test_keister_problem(tb):
    """Test Keister problem setup"""
    tb.inject("""
    from qmcpy import *
    from numpy import *
    """)
    
    tb.execute_cell(2)  # compare function
    tb.execute_cell(4)  # keister problem
    
    keister_problem = tb.ref("keister_problem")
    assert callable(keister_problem)
    
    # Test function structure
    tb.inject("""
    test_distrib = IIDStdUniform(1, seed=7)
    k, cvs, cvmus = keister_problem(test_distrib)
    
    assert k is not None
    assert len(cvs) == 2
    assert len(cvmus) == 2
    assert abs(cvmus[0] - 2/pi) < 1e-10
    assert cvmus[1] == 3/4
    """)

def test_option_problem(tb):
    """Test option pricing problem setup"""
    tb.inject("""
    from qmcpy import *
    from numpy import *
    """)
    
    tb.execute_cell(2)  # compare function
    tb.execute_cell(5)  # option problem
    
    # Check parameters
    call_put = tb.ref("call_put")
    start_price = tb.ref("start_price")
    strike_price = tb.ref("strike_price")
    volatility = tb.ref("volatility")
    interest_rate = tb.ref("interest_rate")
    t_final = tb.ref("t_final")
    
    assert call_put == 'call'
    assert start_price == 100
    assert strike_price == 125
    assert volatility == 0.75
    assert interest_rate == 0.01
    assert t_final == 1
    
    # Check option_problem function
    option_problem = tb.ref("option_problem")
    assert callable(option_problem)

def test_control_variate_structure(tb):
    """Test that control variates are properly structured"""
    tb.inject("""
    from qmcpy import *
    from numpy import *
    """)
    
    tb.execute_cell(2)  # compare function
    
    # Test with a simple problem
    tb.inject("""
    # Create a simple test problem
    def test_problem(discrete_distrib):
        g = CustomFun(Uniform(discrete_distrib, 0, 1), lambda t: t[:, 0])
        cv = CustomFun(Uniform(discrete_distrib, 0, 1), lambda t: t[:, 0]**2)
        return g, [cv], [1/3]
    
    # Test that compare function handles it correctly
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        compare(test_problem, IIDStdUniform(1, seed=7), CubMCCLT, abs_tol=1e-2)
    
    output = f.getvalue()
    assert 'W CV:' in output
    assert 'WO CV:' in output
    assert 'Control variates took' in output
    """)

def test_stopping_criteria_compatibility(tb):
    """Test that different stopping criteria work with control variates"""
    tb.inject("""
    from qmcpy import *
    from numpy import *
    """)
    
    tb.execute_cell(2)  # compare function
    
    # Test multiple stopping criteria
    tb.inject("""
    def simple_problem(discrete_distrib):
        g = CustomFun(Uniform(discrete_distrib, 0, 1), lambda t: t[:, 0])
        cv = CustomFun(Uniform(discrete_distrib, 0, 1), lambda t: t[:, 0])
        return g, [cv], [0.5]
    
    # Test with CubMCCLT
    sc1 = CubMCCLT(simple_problem(IIDStdUniform(1, seed=7))[0], abs_tol=1e-2)
    assert sc1 is not None
    
    # Test with control variates
    g, cvs, cvmus = simple_problem(IIDStdUniform(1, seed=7))
    sc2 = CubMCCLT(g, abs_tol=1e-2, control_variates=cvs, control_variate_means=cvmus)
    assert sc2 is not None
    """)

def test_performance_comparison(tb):
    """Test that control variates actually provide benefit in some cases"""
    tb.inject("""
    from qmcpy import *
    from numpy import *
    import time
    """)
    
    tb.execute_cell(2)  # compare function
    tb.execute_cell(3)  # polynomial problem
    
    # Run a comparison and check that CV uses fewer samples
    tb.inject("""
    # Run polynomial problem comparison
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        compare(poly_problem, IIDStdUniform(3, seed=7), CubMCCLT, abs_tol=1e-2)
    
    output = f.getvalue()
    lines = output.strip().split('\\n')
    
    # Extract sample counts
    for line in lines:
        if 'W CV:' in line:
            parts = line.split()
            idx = parts.index('samples')
            samples_without_cv = float(parts[idx + 1])
        elif 'WO CV:' in line:
            parts = line.split()
            idx = parts.index('samples')
            samples_with_cv = float(parts[idx + 1])
    
    # Control variates should use fewer samples for polynomial problem
    assert samples_with_cv < samples_without_cv
    """)

def test_numerical_stability(tb):
    """Test numerical stability of control variate computations"""
    tb.inject("""
    from qmcpy import *
    from numpy import *
    """)
    
    tb.execute_cell(2)  # compare function
    
    # Test with extreme tolerances
    tb.inject("""
    def stable_problem(discrete_distrib):
        g = CustomFun(Uniform(discrete_distrib, 0, 1), lambda t: t[:, 0] + 1e-10)
        cv = CustomFun(Uniform(discrete_distrib, 0, 1), lambda t: t[:, 0])
        return g, [cv], [0.5]
    
    # Should not crash with very tight tolerance
    try:
        compare(stable_problem, IIDStdUniform(1, seed=7), CubMCCLT, abs_tol=1e-10)
        stable = True
    except:
        stable = False
    
    # May not achieve tolerance but should be stable
    assert stable
    """)

