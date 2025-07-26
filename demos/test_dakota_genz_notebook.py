# test_dakota_genz_notebook.py
import testbook
import pytest
import numpy as np
import pandas as pd

@pytest.fixture(scope="module")
def tb():
    with testbook.testbook('dakota_genz.ipynb', execute=False) as tb:
        yield tb

def test_imports_and_setup(tb):
    """Test imports and matplotlib style setup"""
    tb.execute_cell(0)  # Imports and style
    
    # Check imports
    assert 'numpy' in tb.ref_globals
    assert 'qmcpy' in tb.ref_globals
    assert 'pd' in tb.ref_globals
    assert 'pyplot' in tb.ref_globals
    
    # Check that qmcpy style was applied
    tb.inject("assert 'qmcpy matplotlib style applied' in _")  # Check output

def test_dimension_and_sample_arrays(tb):
    """Test dimension and sample size arrays"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(1)  # arrays setup
    
    kinds_func = tb.ref("kinds_func")
    kinds_coeff = tb.ref("kinds_coeff")
    ds = tb.ref("ds")
    ns = tb.ref("ns")
    
    # Check arrays
    assert kinds_func == ['oscillatory', 'corner-peak']
    assert kinds_coeff == [1, 2, 3]
    assert len(ds) == 8
    assert ds[0] == 1
    assert ds[-1] == 128
    assert len(ns) == 12
    assert ns[0] == 128
    assert ns[-1] == 524288

def test_reference_solutions_computation(tb):
    """Test reference solutions computation"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(1)  # arrays
    
    # Execute with a smaller sample size for testing
    tb.inject("""
    # Override with smaller sizes for testing
    ds = 2**np.arange(3)  # Just [1, 2, 4]
    x_full = DigitalNetB2(ds.max(), seed=7).gen_samples(2**10)  # Smaller sample
    """)
    
    tb.execute_cell(2)  # reference solutions
    
    ref_sols = tb.ref("ref_sols")
    
    # Check DataFrame structure
    assert isinstance(ref_sols, pd.DataFrame)
    assert 'd' == ref_sols.index.name
    assert len(ref_sols.columns) == 6  # 2 functions × 3 coefficients
    
    # Check column names
    expected_cols = ['oscillatory.1', 'oscillatory.2', 'oscillatory.3',
                     'corner-peak.1', 'corner-peak.2', 'corner-peak.3']
    assert all(col in ref_sols.columns for col in expected_cols)

def test_genz_function_types(tb):
    """Test different Genz function types"""
    tb.inject("""
    from qmcpy import *
    import numpy as np
    """)
    
    # Test oscillatory function
    tb.inject("""
    genz_osc = Genz(IIDStdUniform(2), kind_func='oscillatory', kind_coeff=1)
    x_test = np.array([[0.5, 0.5]])
    y_osc = genz_osc.f(x_test)
    assert y_osc.shape == (1,)
    assert np.isfinite(y_osc[0])
    """)
    
    # Test corner-peak function
    tb.inject("""
    genz_cp = Genz(IIDStdUniform(2), kind_func='corner-peak', kind_coeff=1)
    y_cp = genz_cp.f(x_test)
    assert y_cp.shape == (1,)
    assert np.isfinite(y_cp[0])
    assert y_cp[0] >= 0  # Corner peak should be non-negative
    """)

def test_dakota_file_handling(tb):
    """Test Dakota file loading (expected to fail without file)"""
    tb.execute_cell(0)  # imports
    tb.execute_cell(1)  # arrays
    tb.execute_cell(2)  # reference solutions
    
    # This should raise FileNotFoundError
    with pytest.raises(Exception) as exc_info:
        tb.execute_cell(3)  # Dakota file loading
    
    assert "FileNotFoundError" in str(exc_info.typename)

def test_point_generation_methods(tb):
    """Test different point generation methods"""
    tb.inject("""
    from qmcpy import *
    import numpy as np
    
    # Create mock Dakota data
    n_max, d_max = 1024, 8
    x_full_dakota = np.random.rand(n_max, d_max)
    """)
    
    tb.execute_cell(4)  # Point generation
    
    pts = tb.ref("pts")
    
    # Check all methods are present
    expected_methods = [
        'IID Standard Uniform',
        'Lattice (random shift)',
        'Digital Net (random scramble + shift)',
        'Halton (not random, not general)',
        'Halton (not random, general)',
        'Halton (random, not general)',
        'Halton (random, general)',
        'Halton (Dakota)'
    ]
    
    assert all(method in pts for method in expected_methods)
    
    # Check dimensions
    for method, points in pts.items():
        assert points.shape == (1024, 8), f"{method} has wrong shape"
        assert np.all((points >= 0) & (points <= 1)), f"{method} has values outside [0,1]"

def test_halton_configurations(tb):
    """Test different Halton sequence configurations"""
    tb.inject("""
    from qmcpy import *
    import numpy as np
    
    d = 3
    n = 100
    """)
    
    # Test different Halton configurations
    tb.inject("""
    # Not random, not general
    h1 = Halton(d, randomize=False, generalize=False)
    x1 = h1.gen_samples(n, warn=False)
    assert x1.shape == (n, d)
    
    # Random, general
    h2 = Halton(d, randomize=True, generalize=True)
    x2 = h2.gen_samples(n)
    assert x2.shape == (n, d)
    
    # Points should be different due to randomization
    assert not np.allclose(x1, x2)
    """)

def test_genz_function_evaluation(tb):
    """Test Genz function evaluation on different distributions"""
    tb.inject("""
    from qmcpy import *
    import numpy as np
    
    d = 2
    n = 100
    """)
    
    # Test with different distributions
    tb.inject("""
    distributions = [
        IIDStdUniform(d, seed=7),
        Lattice(d, seed=7),
        DigitalNetB2(d, seed=7)
    ]
    
    for dist in distributions:
        genz = Genz(dist, kind_func='oscillatory', kind_coeff=1)
        x = dist.gen_samples(n)
        y = genz.f(x)
        
        assert y.shape == (n,)
        assert np.all(np.isfinite(y))
        assert y.dtype == np.float64
    """)

def test_error_computation(tb):
    """Test error computation logic"""
    tb.inject("""
    import numpy as np
    
    # Test error calculation
    true_val = 0.5
    estimates = np.array([0.45, 0.48, 0.52, 0.51])
    errors = np.abs(estimates - true_val)
    
    assert len(errors) == len(estimates)
    assert np.all(errors >= 0)
    assert errors[0] == 0.05
    assert errors[2] == 0.02
    """)

def test_plotting_setup(tb):
    """Test that plotting would work (without actually plotting)"""
    tb.inject("""
    from matplotlib import pyplot
    import numpy as np
    
    # Test plot setup
    fig, ax = pyplot.subplots(2, 2, figsize=(10, 10))
    assert ax.shape == (2, 2)
    
    # Test log scale setup
    ax[0, 0].set_xscale('log', base=2)
    ax[0, 0].set_yscale('log', base=10)
    
    # Close figure to avoid memory issues
    pyplot.close(fig)
    """)

def test_dataframe_operations(tb):
    """Test pandas DataFrame operations used in the notebook"""
    tb.inject("""
    import pandas as pd
    import numpy as np
    
    # Create test DataFrame similar to ref_sols
    data = {
        'test.1': np.random.rand(3),
        'test.2': np.random.rand(3)
    }
    df = pd.DataFrame(data)
    df['d'] = [1, 2, 4]
    df.set_index('d', inplace=True)
    
    # Test accessing by index
    val = df.loc[2, 'test.1']
    assert isinstance(val, (float, np.floating))
    
    # Test DataFrame structure
    assert df.index.name == 'd'
    assert len(df.columns) == 2
    """)

# Integration test
def test_small_scale_integration(tb):
    """Run a small-scale version of the full computation"""
    tb.inject("""
    from qmcpy import *
    import numpy as np
    import pandas as pd
    
    # Small scale parameters
    ds = np.array([1, 2])
    ns = np.array([128, 256])
    kinds_func = ['oscillatory']
    kinds_coeff = [1]
    
    # Generate reference solution
    x_full = DigitalNetB2(2, seed=7).gen_samples(512)
    ref_sols = {}
    
    for kind_func in kinds_func:
        for kind_coeff in kinds_coeff:
            tag = f'{kind_func}.{kind_coeff}'
            mu_hats = np.zeros(len(ds))
            for j, d in enumerate(ds):
                genz = Genz(IIDStdUniform(d), kind_func=kind_func, kind_coeff=kind_coeff)
                y = genz.f(x_full[:, :d])
                mu_hats[j] = y.mean()
            ref_sols[tag] = mu_hats
    
    ref_sols_df = pd.DataFrame(ref_sols)
    
    # Check results
    assert len(ref_sols_df) == 2
    assert 'oscillatory.1' in ref_sols_df.columns
    assert np.all(np.isfinite(ref_sols_df.values))
    """)
