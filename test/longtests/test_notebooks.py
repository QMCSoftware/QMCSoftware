import testbook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  # Import the os module

# --- Configuration ---
# Define the default path to your notebook
DEFAULT_NOTEBOOK_PATH = "../../demos/dakota_genz.ipynb"

# Get the notebook path from the 'NOTEBOOK_PATH' environment variable.
# If the environment variable is not set, use the DEFAULT_NOTEBOOK_PATH.
NOTEBOOK_PATH = os.environ.get('NOTEBOOK_PATH', DEFAULT_NOTEBOOK_PATH)
# --- End Configuration ---

# Check if the notebook exists at the determined path before running tests (optional but good practice)
if not os.path.exists(NOTEBOOK_PATH):
    print(f"Error: Notebook not found at path: {NOTEBOOK_PATH}")
    print("Please ensure the path is correct or set the NOTEBOOK_PATH environment variable.")
    # You might want to exit or raise an exception here depending on your CI/CD setup
    # For now, we'll let testbook handle the error, but this provides a clearer message.

# Use the NOTEBOOK_PATH variable in the decorator.
# 'execute=True' runs all cells. Adjust if you need finer control.
@testbook.test(NOTEBOOK_PATH, execute=True)
def test_cell_3_setup_and_imports(tb):
    """Tests the setup and import cell (Cell 3)."""
    try:
        tb.inject("import gdown")
        tb.inject("import qmcpy")
    except ImportError:
        assert False, "gdown or qmcpy could not be imported"

    notebook_plt = tb.ref("plt")
    assert notebook_plt.rcParams["text.usetex"] is True, "text.usetex should be True"
    assert notebook_plt.rcParams["font.family"] == ["serif"], "font.family should be serif"

@testbook.test(NOTEBOOK_PATH, execute=True)
def test_cell_5_variable_definitions(tb):
    """Tests the variable definition cell (Cell 5)."""
    kinds_func = tb.ref("kinds_func")
    kinds_coeff = tb.ref("kinds_coeff")
    ds_arr = tb.ref("ds")
    ns_arr = tb.ref("ns")

    assert kinds_func == ['oscillatory', 'corner-peak']
    assert kinds_coeff == [1, 2, 3]

    assert isinstance(ds_arr, np.ndarray)
    assert np.array_equal(ds_arr, 2**np.arange(8))

    assert isinstance(ns_arr, np.ndarray)
    assert np.array_equal(ns_arr, 2**np.arange(7,19))

@testbook.test(NOTEBOOK_PATH, execute=True)
def test_cell_6_ref_sols_dataframe(tb):
    """Tests the reference solutions DataFrame generation (Cell 6)."""
    ref_sols_df = tb.ref("ref_sols")
    ds_arr = tb.ref("ds")

    assert isinstance(ref_sols_df, pd.DataFrame)
    assert len(ref_sols_df.columns) == 6
    assert list(ref_sols_df.index) == list(ds_arr)
    assert ref_sols_df.shape == (len(ds_arr), 6)

    # Note: These specific values should be verified against a known correct run.
    # Floating point comparisons should use np.isclose for robustness.
    assert np.isclose(ref_sols_df.loc[1, 'oscillatory.1'], -0.217229, atol=1e-6)
    assert np.isclose(ref_sols_df.loc[128, 'corner-peak.3'], 0.000003, atol=1e-7)

@testbook.test(NOTEBOOK_PATH, execute=True)
def test_cell_7_load_x_full_dakota(tb):
    """Tests loading the Dakota data (Cell 7)."""
    x_full_dakota = tb.ref("x_full_dakota")
    assert isinstance(x_full_dakota, np.ndarray)
    assert x_full_dakota.size > 0
    # Add a shape check if you know the expected dimensions, e.g.,
    # assert x_full_dakota.shape == (262144, 128) # Replace with actual shape

@testbook.test(NOTEBOOK_PATH, execute=True)
def test_cell_8_pts_dictionary(tb):
    """Tests the 'pts' dictionary creation (Cell 8)."""
    pts_dict = tb.ref("pts")
    n_max = tb.ref("n_max")
    d_max = tb.ref("d_max")

    assert isinstance(pts_dict, dict)
    expected_keys = [
        'IID Standard Uniform', 'Lattice (random shift)',
        'Digital Net (random scramble + shift)', 'Halton (not random, not general)',
        'Halton (not random, general)', 'Halton (random, not general)',
        'Halton (random, general)', 'Halton (Dakota)'
    ]
    assert all(key in pts_dict for key in expected_keys)
    assert pts_dict['IID Standard Uniform'].shape == (n_max, d_max)

@testbook.test(NOTEBOOK_PATH, execute=True)
def test_cell_9_plotting_cell_runs(tb):
    """Tests that the plotting cell (Cell 9) executes without errors."""
    # This test primarily checks for execution errors.
    # It assumes that if the cell runs, it's successful for this level of testing.
    # No direct assertions are needed unless specific variables are created/modified.
    # If the cell runs due to `execute=True`, this test function simply passes.
    # We can add an 'assert True' for clarity or try/except if we only wanted to run this cell.
    assert True
