# test_qmcpy_intro_notebook.py
import testbook
import pytest
import numpy as np

@pytest.fixture(scope="module")
def tb():
    with testbook.testbook('qmcpy_intro.ipynb', execute=False) as tb:
        yield tb

def test_import_methods(tb):
    """Test different import methods"""
    # Test first import method
    tb.execute_cell(0)  # import qmcpy as qp
    
    tb.inject("""
    assert 'qp' in globals()
    assert hasattr(qp, 'name')
    assert hasattr(qp, '__version__')
    assert qp.name == 'qmcpy'
    """)

def test_individual_imports(tb):
    """Test importing individual modules"""
    tb.execute_cell(0)  # Previous import
    tb.execute_cell(1)  # Individual imports
    
    # Check that classes are available
    tb.inject("""
    # These should be available from the imports
    test_classes = [
        'Keister', 'AsianOption', 'EuropeanOption',  # integrands
        'Uniform', 'Gaussian', 'Lebesgue',  # true measures
        'IIDStdUniform', 'Lattice', 'Sobol',  # distributions
        'CubMCCLT', 'CubQMCSobolG'  # stopping criteria
    ]
    
    for cls_name in test_classes:
        assert cls_name in globals() or hasattr(qmcpy, cls_name)
    """)

def test_wildcard_import(tb):
    """Test wildcard import"""
    tb.execute_cell(0)
    tb.execute_cell(1)
    tb.execute_cell(2)  # from qmcpy import *
    
    # Wildcard import should make everything available
    tb.inject("""
    assert 'CustomFun' in globals()
    assert 'Uniform' in globals()
    """)

def test_lattice_generation(tb):
    """Test lattice sequence generation"""
    tb.execute_cell(0)
    tb.execute_cell(1)
    tb.execute_cell(2)
    tb.execute_cell(3)  # Lattice generation
    
    # Check the generated samples
    tb.inject("""
    samples = distribution.gen_samples(n_min=0, n_max=4)
    assert samples.shape == (4, 2)
    assert np.all((samples >= 0) & (samples <= 1))
    """)

def test_function_definition_attempts(tb):
    """Test the function definition process"""
    # Execute imports
    for i in range(4):
        tb.execute_cell(i)
    
    # Import numpy functions
    tb.execute_cell(4)
    
    # Test initial function definition
    tb.execute_cell(5)  # def f(x)
    tb.execute_cell(6)  # test with scalar
    
    # Check scalar evaluation
    tb.inject("""
    result = f(0.01)
    assert isinstance(result, (int, float))
    assert result > 0
    """)

def test_array_function_debugging(tb):
    """Test debugging process for array inputs"""
    # Execute previous cells
    for i in range(7):
        tb.execute_cell(i)
    
    tb.execute_cell(7)  # Array test that fails
    
    # The function returns a scalar instead of array
    tb.inject("""
    x_test = array([[1., 0.], [0., 0.01], [0.04, 0.04]])
    result = f(x_test)
    assert isinstance(result, (int, float))  # Wrong behavior
    """)

def test_corrected_function(tb):
    """Test the corrected function"""
    # Execute up to the corrected function
    for i in range(12):
        tb.execute_cell(i)
    
    tb.execute_cell(12)  # Corrected myfunc
    
    # Test corrected function
    tb.inject("""
    x_test = array([[1., 0.], [0., 0.01], [0.04, 0.04]])
    result = myfunc(x_test)
    assert len(result) == 3
    assert np.all(np.isfinite(result))
    """)

def test_integration_1d(tb):
    """Test 1D integration"""
    # Execute necessary cells
    for i in range(13):
        tb.execute_cell(i)
    
    tb.execute_cell(13)  # 1D integration
    
    # Check integration results
    tb.inject("""
    assert 'solution' in globals()
    assert 'data' in globals()
    assert isinstance(solution, (int, float))
    assert 0.6 < solution < 0.7  # Expected range
    assert hasattr(data, 'n_total')
    """)

def test_integration_accuracy_1d(tb):
    """Test accuracy check for 1D"""
    # Execute previous cells
    for i in range(14):
        tb.execute_cell(i)
    
    tb.execute_cell(14)  # Accuracy check
    
    # Should not raise exception
    assert True

def test_integration_2d(tb):
    """Test 2D integration"""
    # Execute previous cells
    for i in range(15):
        tb.execute_cell(i)
    
    tb.execute_cell(15)  # 2D integration
    
    # Check 2D results
    tb.inject("""
    assert dim == 2
    assert 'solution2' in globals()
    assert 'data2' in globals()
    assert 0.8 < solution2 < 0.85  # Expected range
    """)

def test_integration_accuracy_2d(tb):
    """Test accuracy check for 2D"""
    # Execute all cells
    for i in range(16):
        tb.execute_cell(i)
    
    tb.execute_cell(16)  # 2D accuracy check
    
    # Should not raise exception
    assert True

def test_custom_function_properties(tb):
    """Test custom function integration properties"""
    tb.inject("""
    from qmcpy import *
    import numpy as np
    from numpy.linalg import norm
    
    # Test that CustomFun works with various functions
    def test_func(x):
        return np.sin(x.sum(1))
    
    integrand = CustomFun(
        true_measure=Uniform(IIDStdUniform(3, seed=7)),
        g=test_func
    )
    
    x = integrand.discrete_distrib.gen_samples(100)
    y = integrand.f(x)
    
    assert len(y) == 100
    assert np.all(np.isfinite(y))
    assert y.min() >= -1
    assert y.max() <= 1
    """)

