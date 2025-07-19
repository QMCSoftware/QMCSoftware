# test_lebesgue_integration_notebook.py
import testbook
import pytest
import numpy as np

@pytest.fixture(scope="module")
def tb():
    with testbook.testbook('lebesgue_integration.ipynb', execute=False) as tb:
        yield tb

def test_imports(tb):
    """Test imports"""
    tb.execute_cell(0)  # imports
    
    assert 'qmcpy' in tb.ref_globals
    assert 'numpy' in tb.ref_globals

def test_sample_problem_1_setup(tb):
    """Test sample problem 1 parameters"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(1)  # problem setup
    
    abs_tol = tb.ref("abs_tol")
    dim = tb.ref("dim")
    a = tb.ref("a")
    b = tb.ref("b")
    true_value = tb.ref("true_value")
    
    assert abs_tol == 0.01
    assert dim == 1
    assert a == 0
    assert b == 2
    assert abs(true_value - 8/3) < 1e-10

def test_lebesgue_measure_1d(tb):
    """Test Lebesgue measure integration"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(1)  # setup
    tb.execute_cell(2)  # Lebesgue measure
    
    # Check solution accuracy
    tb.inject("""
    assert 'solution' in globals()
    assert abs(solution - true_value) <= abs_tol
    """)

def test_uniform_measure_1d(tb):
    """Test uniform measure integration"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(1)  # setup
    tb.execute_cell(3)  # Uniform measure
    
    # Check solution accuracy
    tb.inject("""
    assert 'solution' in globals()
    assert abs(solution - true_value) <= abs_tol
    """)

def test_sample_problem_2_setup(tb):
    """Test sample problem 2 parameters"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(4)  # problem 2 setup
    
    # Check multidimensional setup
    tb.inject("""
    assert dim == 2
    assert len(a) == 2
    assert len(b) == 2
    assert np.array_equal(a, np.array([1., 2.]))
    assert np.array_equal(b, np.array([2., 4.]))
    assert abs(true_value - 23.33333) < 0.00001
    """)

def test_lebesgue_measure_2d(tb):
    """Test 2D Lebesgue measure"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(4)  # setup
    tb.execute_cell(5)  # Lebesgue 2D
    
    tb.inject("""
    assert abs(solution - true_value) <= abs_tol
    """)

def test_uniform_measure_2d(tb):
    """Test 2D uniform measure"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(4)  # setup
    tb.execute_cell(6)  # Uniform 2D
    
    tb.inject("""
    assert abs(solution - true_value) <= abs_tol
    """)

def test_sample_problem_3(tb):
    """Test sin(x)/log(x) integral"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(7)  # setup
    tb.execute_cell(8)  # integration
    
    # Check problem 3 specific values
    tb.inject("""
    assert abs_tol == 0.0001
    assert dim == 1
    assert a == 3
    assert b == 5
    assert abs(true_value - (-0.87961)) < 1e-5
    assert abs(solution - true_value) <= abs_tol
    """)

def test_sample_problem_4(tb):
    """Test integral over R^d"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(9)  # setup
    tb.execute_cell(10)  # integration
    
    # Check Gaussian integral
    tb.inject("""
    assert abs_tol == 0.1
    assert dim == 2
    assert abs(true_value - np.pi) < 1e-10
    assert abs(solution - true_value) <= abs_tol
    """)

def test_measure_types(tb):
    """Test different measure types"""
    tb.inject("""
    from qmcpy import *
    import numpy as np
    
    # Test Lebesgue measure
    leb_measure = Lebesgue(Uniform(IIDStdUniform(1, seed=7), 0, 1))
    assert hasattr(leb_measure, 'sampler')
    
    # Test that Lebesgue measure scales correctly
    integrand = CustomFun(
        true_measure=leb_measure,
        g=lambda x: np.ones(x.shape[0])
    )
    x = integrand.discrete_distrib.gen_samples(1000)
    y = integrand.f(x)
    assert abs(y.mean() - 1.0) < 0.1  # Should integrate to length of interval
    """)

def test_custom_functions(tb):
    """Test custom function integration"""
    tb.inject("""
    from qmcpy import *
    import numpy as np
    
    # Test various custom functions
    def f1(x): return x.sum(1)**2
    def f2(x): return np.exp(-x.sum(1))
    def f3(x): return np.sin(x.sum(1))
    
    for f in [f1, f2, f3]:
        integrand = CustomFun(
            true_measure=Uniform(IIDStdUniform(2, seed=7)),
            g=f
        )
        x = integrand.discrete_distrib.gen_samples(100)
        y = integrand.f(x)
        assert len(y) == 100
        assert np.all(np.isfinite(y))
    """)

def test_different_distributions(tb):
    """Test with different discrete distributions"""
    tb.inject("""
    from qmcpy import *
    import numpy as np
    
    distributions = [
        IIDStdUniform(2, seed=7),
        Lattice(2, seed=7),
        DigitalNetB2(2, seed=7),
        Halton(2, seed=7)
    ]
    
    for dist in distributions:
        integrand = CustomFun(
            true_measure=Lebesgue(Uniform(dist, 0, 1)),
            g=lambda x: x.sum(1)
        )
        sol, data = CubQMCCLT(integrand, abs_tol=0.1).integrate()
        # Integral of x+y over [0,1]^2 = 1
        assert abs(sol - 1.0) < 0.1
