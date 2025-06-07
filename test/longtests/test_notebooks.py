# test_notebooks.py

import pytest
from testbook import testbook
import numpy as np

# The @testbook decorator targets the notebook file and enables kernel interaction
@testbook("pricing_asian_options.ipynb", execute=True)
def test_notebook_execution_and_setup(tb):
    """
    Tests that the notebook executes without errors up to the European Option
    section and that initial parameters are set correctly.
    """
    # Check if the notebook executed without raising an exception
    assert tb.cell_executed_count > 0

    # Use tb.ref to get a reference to a variable in the notebook's namespace
    init_price = tb.ref("initPrice")
    vol = tb.ref("vol")
    sample_size = tb.ref("sampleSize")

    # Assert that the initial parameters are correct
    assert init_price == 120
    assert vol == 0.5
    assert sample_size == 10**6
    print("\nPASSED: Notebook setup and initial cells executed successfully.")


@testbook("pricing_asian_options.ipynb", execute=True)
def test_european_option_pricing(tb):
    """
    Tests the European Call Option pricing cell.
    It verifies that the Monte Carlo estimation is close to the exact value.
    """
    euro_call = tb.ref("EuroCall")
    estimated_price = tb.ref("y").mean()
    exact_price = euro_call.get_exact_value()

    # The notebook output for the estimated price is 8.2741
    assert estimated_price == pytest.approx(8.2741, abs=1e-4)
    # The estimation should be reasonably close to the exact value
    assert estimated_price == pytest.approx(exact_price, abs=0.01)
    print("\nPASSED: European Option pricing is correct.")


@testbook("pricing_asian_options.ipynb", execute=True)
def test_arithmetic_mean_options(tb):
    """
    Tests the Arithmetic Mean Asian Option pricing for both call and put.
    It also checks the daily monitoring case.
    """
    # --- Test 1: Initial Arithmetic Call Option ---
    # The notebook calculates this in cell 5
    arith_call_price = tb.ref("y").mean()
    assert arith_call_price == pytest.approx(3.3857, abs=1e-4)

    # --- Test 2: Arithmetic Put Option ---
    # We need to execute the next cell (index 7) to get the put price
    tb.execute_cell(7)
    arith_put_price = tb.ref("y").mean()
    assert arith_put_price == pytest.approx(13.0448, abs=1e-4)

    # --- Test 3: Daily Monitoring Call Option ---
    # Execute the cell for daily monitoring (index 9)
    tb.execute_cell(9)
    daily_monitoring_price = tb.ref("y").mean()
    assert daily_monitoring_price == pytest.approx(3.3948, abs=1e-4)
    print("\nPASSED: Arithmetic Mean Option pricing is correct.")


@testbook("pricing_asian_options.ipynb", execute=True)
def test_geometric_mean_options(tb):
    """
    Tests the Geometric Mean Asian Option pricing for both put and call.
    Verifies the relationship between arithmetic and geometric mean option prices.
    """
    # --- Get Arithmetic Price for comparison ---
    arith_call_price = 3.3857  # from previous test
    arith_put_price = 13.0448  # from previous test

    # --- Test 1: Geometric Put Option ---
    # Execute the geometric put cell (index 11)
    tb.execute_cell(11)
    geo_put_price = tb.ref("y").mean()
    assert geo_put_price == pytest.approx(13.4415, abs=1e-4)

    # --- Test 2: Geometric Call Option ---
    # Execute the geometric call cell (index 12)
    tb.execute_cell(12)
    geo_call_price = tb.ref("y").mean()
    assert geo_call_price == pytest.approx(3.1267, abs=1e-4)

    # As per the notebook, GeoMeanCall <= ArithMeanCall
    assert geo_call_price < arith_call_price
    # As per the notebook, GeoMeanPut >= ArithMeanPut
    assert geo_put_price > arith_put_price
    print("\nPASSED: Geometric Mean Option pricing is correct.")


@testbook("pricing_asian_options.ipynb", execute=True)
def test_barrier_option(tb):
    """
    Tests the Barrier Option pricing.
    Verifies that the Up-and-In call price is less than the standard European
    call price.
    """
    # Get European price for comparison
    euro_call_price = tb.ref("EuroCall").get_exact_value()

    # Execute the barrier option cell (index 14)
    tb.execute_cell(14)
    barrier_price = tb.ref("y").mean()

    assert barrier_price == pytest.approx(7.4060, abs=1e-4)
    # Price should be less than the European option, as the barrier adds a condition
    assert barrier_price < euro_call_price
    print("\nPASSED: Barrier Option pricing is correct.")


@testbook("pricing_asian_options.ipynb")
def test_lookback_option_fixed(tb):
    """
    Tests the Lookback Option cell after fixing it.
    The original cell has several errors:
    1. It doesn't pass a sampler to the LookBackOption constructor.
    2. It tries to generate samples from the wrong object.
    3. It prints the wrong sample size.
    We use tb.inject() to replace the cell's content with corrected code.
    """
    # Execute all cells up to the problematic one
    tb.execute_cell(list(range(15)))

    # This is the corrected code for the Lookback Option cell
    corrected_code = """
import qmcpy as qp
vol = 0.5
initPrice = 120
interest = 0.02
tfinal = 1/4
d = 12
sampleSize_lookback = 2**12

# Correctly instantiate the sampler and the option
lookback_sampler = qp.IIDStdUniform(dimension=d, seed=7)
LookCall = qp.LookBackOption(lookback_sampler, volatility=vol,
                             start_price=initPrice, interest_rate=interest,
                             t_final=tfinal, call_put='call')

# Generate samples and compute the payoff
x = LookCall.discrete_distrib.gen_samples(sampleSize_lookback)
y = LookCall.f(x)

# Print the correct information
print(f"After generate {sampleSize_lookback} iid points, the price of this Lookback Call Option is {y.mean():.4f}")
"""
    # Inject and execute the corrected code in place of the original cell (index 16)
    tb.inject(corrected_code, cell_index=16)
    tb.execute_cell(16)

    lookback_price = tb.ref("y").mean()

    # The corrected code with a fixed seed gives a deterministic result.
    # This value was pre-calculated by running the corrected snippet.
    assert lookback_price == pytest.approx(13.9131, abs=1e-4)
    print("\nPASSED: Lookback Option pricing is correct after fixing the cell.")
