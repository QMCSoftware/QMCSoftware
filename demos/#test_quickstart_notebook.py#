# test_quickstart_notebook.py
import testbook
import pytest
import numpy as np

@pytest.fixture(scope="module")
def tb():
    with testbook.testbook('quickstart.ipynb', execute=False) as tb:
        yield tb

def test_keister_function_definition(tb):
    """Test Keister function implementation"""
    tb.execute_cell(0)  # Keister function definition
    
    # Test the function
    tb.inject("""
    # Test with single sample
    x = np.array([[1.0, 0.0]])
    k = keister(x)
    assert k.shape == (1,)
    assert k[0] == np.pi
    
    # Test with multiple samples
    x = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    k = keister(x)
    assert k.shape == (3,)
    assert np.allclose(k[0], np.pi)
    assert np.allclose(k[1], np.pi)
    assert np.allclose(k[2], np.pi)
    """)

def test_qmcpy_setup(tb):
    """Test QMCPy setup for Keister example"""
    tb.execute_cell(0)  # Keister function
    tb.execute_cell(1)  # QMCPy setup
    
    # Check objects were created
    tb.inject("""
    assert d == 2
    assert isinstance(discrete_distrib, qmcpy.Lattice)
    assert discrete_distrib.d == 2
    
    assert isinstance(true_measure, qmcpy.Gaussian)
    assert true_measure.mean == 0
    assert true_measure.covariance == 0.5
    
    assert isinstance(integrand, qmcpy.CustomFun)
    assert integrand.g == keister
    
    assert isinstance(stopping_criterion, qmcpy.CubQMCLatticeG)
    assert stopping_criterion.abs_tol == 1e-3
    """)

def test_integration(tb):
    """Test integration execution"""
    tb.execute_cell(0)  # Keister function
    tb.execute_cell(1)  # Setup
    tb.execute_cell(2)  # Integration
    
    # Check results
    tb.inject("""
    assert 'solution' in globals()
    assert 'data' in globals()
    
    # Solution should be close to 1.808 for d=2
    assert 1.7 < solution < 1.9
    
    # Check data properties
    assert hasattr(data, 'n_total')
    assert data.n_total > 0
    assert hasattr(data, 'time_integrate')
    assert data.time_integrate > 0
    
    # Check bounds
    assert hasattr(data, 'comb_bound_low')
    assert hasattr(data, 'comb_bound_high')
    assert data.comb_bound_low <= solution <= data.comb_bound_high
    """)

def test_data_output_structure(tb):
    """Test the structure of output data"""
    tb.execute_cell(0)
    tb.execute_cell(1)
    tb.execute_cell(2)
    
    tb.inject("""
    # Check data object attributes
    assert hasattr(data, 'solution')
    assert hasattr(data, 'n_total')
    assert hasattr(data, 'time_integrate')
    assert hasattr(data, 'stopping_crit')
    assert hasattr(data, 'integrand')
    
    # Check stopping criterion info
    sc = data.stopping_crit
    assert sc.abs_tol == 1e-3
    assert sc.rel_tol == 0
    
    # Check integrand info
    integ = data.integrand
    assert hasattr(integ, 'true_measure')
    assert hasattr(integ, 'discrete_distrib')
    """)

def test_keister_properties(tb):
    """Test mathematical properties of Keister function"""
    tb.inject("""
    import numpy as np
    import qmcpy
    
    # Test Keister at origin
    k_origin = keister(np.array([[0.0, 0.0]]))
    assert np.allclose(k_origin[0], np.pi)
    
    # Test symmetry
    x1 = np.array([[1.0, 0.0]])
    x2 = np.array([[0.0, 1.0]])
    x3 = np.array([[-1.0, 0.0]])
    x4 = np.array([[0.0, -1.0]])
    
    k1, k2, k3, k4 = keister(x1), keister(x2), keister(x3), keister(x4)
    
    # All should be equal due to radial symmetry
    assert np.allclose(k1, k2)
    assert np.allclose(k1, k3)
    assert np.allclose(k1, k4)
    """)

def test_different_dimensions(tb):
    """Test Keister integration in different dimensions"""
    tb.inject("""
    import qmcpy
    
    # Test different dimensions
    for d in [1, 3, 5]:
        discrete_distrib = qmcpy.Lattice(dimension=d)
        true_measure = qmcpy.Gaussian(discrete_distrib, mean=0, covariance=0.5)
        integrand = qmcpy.CustomFun(true_measure, keister)
        sc = qmcpy.CubQMCLatticeG(integrand=integrand, abs_tol=0.1)
        
        sol, _ = sc.integrate()
        assert isinstance(sol, (int, float))
        assert sol > 0
    """)

def test_lattice_properties(tb):
    """Test lattice sequence properties"""
    tb.inject("""
    import qmcpy
    import numpy as np
    
    # Create lattice
    lat = qmcpy.Lattice(dimension=2, seed=123)
    
    # Generate samples
    x = lat.gen_samples(100)
    
    assert x.shape == (100, 2)
    assert np.all((x >= 0) & (x <= 1))
    
    # Check randomization
    lat2 = qmcpy.Lattice(dimension=2, seed=456)
    x2 = lat2.gen_samples(100)
    
    assert not np.allclose(x, x2)  # Different seeds should give different shifts
    """)

