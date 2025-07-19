# test_integration_examples_notebook.py
import testbook
import pytest
import numpy as np

@pytest.fixture(scope="module")
def tb():
    with testbook.testbook('integration_examples.ipynb', execute=False) as tb:
        yield tb

def test_imports(tb):
    """Test that imports work correctly"""
    tb.execute_cell(0)  # Import cell
    
    # Check imports
    assert 'qmcpy' in tb.ref_globals
    assert 'arange' in tb.ref_globals

def test_keister_example(tb):
    """Test Keister example execution"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(1)  # Keister example
    
    # Check data object was created
    tb.inject("assert 'data' in globals()")
    tb.inject("assert 'solution' in globals()")
    
    # Check solution is reasonable
    tb.inject("""
    assert isinstance(solution, (int, float))
    assert 0 < solution < 10  # Reasonable bounds for Keister
    assert hasattr(data, 'n_total')
    assert data.n_total > 0
    """)

def test_asian_option_single_level(tb):
    """Test Asian option single level"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(2)  # Asian option single level
    
    # Check that solution was computed
    tb.inject("""
    assert isinstance(solution, (int, float))
    assert solution > 0  # Option price should be positive
    assert hasattr(data, 'time_integrate')
    assert data.time_integrate > 0
    """)

def test_asian_option_multi_level(tb):
    """Test Asian option multi-level"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(3)  # Asian option multi-level
    
    # Check multi-level specific attributes
    tb.inject("""
    assert hasattr(data, 'levels')
    assert data.levels > 1  # Should use multiple levels
    assert hasattr(data.integrand, 'multilevel_dims')
    assert np.array_equal(data.integrand.multilevel_dims, [4, 8, 16])
    """)

def test_bayesian_cubature_lattice(tb):
    """Test Bayesian cubature with lattice"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(4)  # Bayesian cubature with lattice
    
    # Check results
    tb.inject("""
    assert 'solution' in globals()
    assert 'data' in globals()
    assert isinstance(solution, (int, float))
    assert hasattr(data, 'comb_bound_low')
    assert hasattr(data, 'comb_bound_high')
    assert data.comb_bound_low <= solution <= data.comb_bound_high
    """)

def test_bayesian_cubature_sobol(tb):
    """Test Bayesian cubature with Sobol"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(5)  # Bayesian cubature with Sobol
    
    # Check Sobol-specific attributes
    tb.inject("""
    assert isinstance(data.integrand.discrete_distrib, Sobol)
    assert data.integrand.discrete_distrib.graycode == False
    assert data.n_total > 0
    """)

def test_integrand_properties(tb):
    """Test integrand properties across examples"""
    tb.inject("""
    from qmcpy import *
    
    # Test Keister integrand
    k = Keister(Sobol(dimension=3, seed=7))
    assert hasattr(k, 'true_measure')
    assert hasattr(k, 'discrete_distrib')
    assert k.d == 3
    
    # Test Asian option
    ao = AsianOption(
        sampler=IIDStdUniform(dimension=4, seed=7),
        volatility=0.5,
        start_price=30,
        strike_price=25
    )
    assert ao.volatility == 0.5
    assert ao.start_price == 30
    assert ao.strike_price == 25
    """)

def test_stopping_criteria(tb):
    """Test different stopping criteria"""
    tb.inject("""
    from qmcpy import *
    
    # Test CubQMCSobolG
    integrand = Keister(Sobol(dimension=2, seed=7))
    sc1 = CubQMCSobolG(integrand, abs_tol=0.1)
    assert sc1.abs_tol == 0.1
    assert sc1.n_max == 2**35
    
    # Test CubMCCLT
    integrand2 = AsianOption(IIDStdUniform(dimension=4, seed=7))
    sc2 = CubMCCLT(integrand2, abs_tol=0.05)
    assert sc2.abs_tol == 0.05
    assert sc2.alpha == 0.01
    
    # Test Bayesian criteria
    integrand3 = Keister(Lattice(dimension=2))
    sc3 = CubBayesLatticeG(integrand3, abs_tol=0.01)
    assert sc3.abs_tol == 0.01
    assert hasattr(sc3, 'order')
    """)

def test_brownian_motion_measure(tb):
    """Test Brownian motion true measure"""
    tb.inject("""
    from qmcpy import *
    import numpy as np
    
    # Create Asian option with Brownian motion
    ao = AsianOption(
        sampler=IIDStdUniform(dimension=16, seed=7),
        volatility=0.5,
        start_price=30,
        strike_price=25,
        interest_rate=0.01,
        mean_type='arithmetic'
    )
    
    # Check Brownian motion properties
    bm = ao.true_measure
    assert hasattr(bm, 'time_vec')
    assert hasattr(bm, 'drift')
    assert hasattr(bm, 'covariance')
    assert len(bm.time_vec) == 16
    """)

def test_solution_consistency(tb):
    """Test that solutions are consistent"""
    tb.inject("""
    from qmcpy import *
    
    # Run same problem twice with same seed
    k1 = Keister(Sobol(dimension=3, seed=123))
    sol1, _ = CubQMCSobolG(k1, abs_tol=0.1).integrate()
    
    k2 = Keister(Sobol(dimension=3, seed=123))
    sol2, _ = CubQMCSobolG(k2, abs_tol=0.1).integrate()
    
    assert abs(sol1 - sol2) < 1e-10  # Should be identical
    """)

