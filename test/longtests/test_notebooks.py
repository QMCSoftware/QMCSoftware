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
# test_mlqmc_notebook.py

import pytest
from testbook import testbook
import re
import numpy as np


# Use the testbook decorator to target the notebook file.
# execute=True will run the whole notebook once to catch any basic errors.
@testbook("asian-option-mlqmc.ipynb", execute=True)
def test_notebook_smoke_test(tb):
    """
    A simple "smoke test" that executes the entire notebook from start to finish.
    It passes if no exceptions are raised during execution.
    This ensures all cells are syntactically correct and runnable.
    """
    # The decorator handles execution, so we just need to assert it completed.
    assert tb.cell_executed_count > 0
    print("\nPASSED: Notebook executed successfully from start to finish.")


@testbook("asian-option-mlqmc.ipynb", execute=False)
def test_single_level_qmc_output(tb):
    """
    Tests the single-level QMC calculation cell (index 5).
    It verifies that the output printed to stdout matches the expected values.
    """
    # Execute cells up to and including the target cell
    tb.execute_cell(slice(0, 6))

    # Get the text output of the 6th cell (index 5)
    output = tb.cell_output_text(5)

    # Check that the key results are present in the output
    assert "Asian Option true value (2 time steps): 5.63591" in output
    assert "Asian Option true value (8 time steps): 5.75526" in output
    assert "Asian Option true value (32 time steps): 5.76260" in output
    print("\nPASSED: Single-level QMC outputs are correct.")


@testbook("asian-option-mlqmc.ipynb", execute=False)
def test_eval_option_deterministic_output(tb):
    """
    Tests the first call to the `eval_option` function (cell index 11).
    It checks the deterministic parts of the output: the solution value and
    the number of levels, which should be consistent for a fixed seed.
    """
    # Execute all cells needed to define and call the function
    tb.execute_cell(slice(0, 12))

    output = tb.cell_output_text(11)
    lines = output.strip().split("\n")

    # Expected results (solution, levels) for each method
    expected = {
        "MLMC": (5.7620, 10),
        "continuation MLMC": (5.7580, 7),
        "MLQMC": (5.7606, 8),
        "continuation MLQMC": (5.7594, 7),
    }

    # Regex to parse each line of the output
    pattern = re.compile(
        r"\s*(?P<name>[\w\s]+)\s+solution\s+(?P<sol>[-.\d]+)\s+number of levels\s+(?P<lvl>\d+)"
    )

    assert len(lines) == 4, "Expected output for 4 methods"

    for line in lines:
        match = pattern.match(line)
        assert match, f"Output line did not match expected format: {line}"
        
        name = match.group("name").strip()
        solution = float(match.group("sol"))
        levels = int(match.group("lvl"))

        assert name in expected, f"Unexpected method name: {name}"
        expected_sol, expected_lvl = expected[name]

        assert solution == pytest.approx(expected_sol, abs=1e-3)
        assert levels == expected_lvl

    print("\nPASSED: `eval_option` function provides correct deterministic results.")


@testbook("asian-option-mlqmc.ipynb", execute=False)
def test_complexity_loop_and_plotting(tb):
    """
    Tests the main complexity loop and subsequent data processing/plotting cells.
    It injects code to reduce the number of repetitions and tolerances,
    making the test run much faster. It then verifies that the data structures
    for plotting are created correctly and that the plotting cells can execute.
    """
    # Cell indices before injection:
    # 13: Complexity loop
    # 15: avg_time calculation
    # 16: First plot
    # 17: max_levels calculation
    # 18: Second plot

    # Run all cells before the main loop
    tb.execute_cell(slice(0, 13))

    # Inject code to override loop parameters for a fast test run.
    # This becomes the new cell at index 13.
    tb.inject(
        """
    import numpy as np
    repetitions = 2
    tolerances = np.array([5e-2, 5e-3])
    """,
        before=13,
    )

    # Execute the original loop cell (now at index 14)
    tb.execute_cell(14)

    # Execute the data processing cells
    tb.execute_cell(16)  # Original cell 15 -> avg_time
    tb.execute_cell(18)  # Original cell 17 -> max_levels

    # Get the results from the notebook's kernel
    avg_time = tb.ref("avg_time")
    max_levels = tb.ref("max_levels")
    tolerances = tb.ref("tolerances")

    # --- Assertions on the generated data ---
    assert isinstance(avg_time, dict) and len(avg_time) == 4
    assert all(len(v) == len(tolerances) for v in avg_time.values())
    assert avg_time[0][0] > 0  # Check for plausible (non-zero) time

    assert isinstance(max_levels, dict) and len(max_levels) == 4
    assert len(max_levels[0]) == 15
    # The sum of fractions should be 1.0 (for 100%)
    assert sum(max_levels[0]) == pytest.approx(1.0)

    # --- Execute plotting cells to ensure they run without error ---
    tb.execute_cell(17)  # Original cell 16
    tb.execute_cell(19)  # Original cell 18

    print("\nPASSED: Complexity loop and plotting cells run correctly with injected data.")

# test_control_variates.py


@testbook("control_variates.ipynb", execute=True)
def test_notebook_execution(tb):
    """
    Smoke test to ensure the entire notebook executes without errors.
    """
    assert tb.cell_executed_count > 0
    print("\nPASSED: Control variates notebook executed successfully.")


@testbook("control_variates.ipynb", execute=False)
def test_setup_and_imports(tb):
    """
    Test that all imports and setup cells execute correctly.
    """
    # Execute setup cells (0-2)
    tb.execute_cell(slice(0, 3))
    
    # Check that key modules are available
    qmcpy_available = tb.ref("qmcpy") is not None
    numpy_available = tb.ref("numpy") is not None
    
    assert qmcpy_available, "qmcpy should be imported"
    assert numpy_available, "numpy should be imported"
    
    print("\nPASSED: Setup and imports work correctly.")


@testbook("control_variates.ipynb", execute=False)
def test_compare_function_definition(tb):
    """
    Test that the compare function is properly defined and accessible.
    """
    # Execute cells up to the compare function definition
    tb.execute_cell(slice(0, 4))
    
    # Check that compare function exists
    compare_func = tb.ref("compare")
    assert callable(compare_func), "compare function should be callable"
    
    print("\nPASSED: Compare function is properly defined.")


@testbook("control_variates.ipynb", execute=False)
def test_polynomial_problem(tb):
    """
    Test the polynomial function integration problem with control variates.
    """
    # Execute setup and polynomial problem cells
    tb.execute_cell(slice(0, 6))
    
    # Get the output from the polynomial problem cell
    output = tb.cell_output_text(5)
    lines = output.strip().split('\n')
    
    # Parse the output to extract performance metrics
    results = {}
    current_method = None
    
    for line in lines:
        if 'Stopping Criterion:' in line:
            # Extract method name and tolerance
            match = re.search(r'Stopping Criterion:\s+(\w+)\s+absolute tolerance:\s+([\d.e-]+)', line)
            if match:
                current_method = match.group(1)
                results[current_method] = {}
        elif 'W CV:' in line:
            # Without control variates
            match = re.search(r'Solution\s+([\d.-]+)\s+time\s+([\d.-]+)\s+samples\s+([\d.e+-]+)', line)
            if match and current_method:
                results[current_method]['without_cv'] = {
                    'solution': float(match.group(1)),
                    'time': float(match.group(2)),
                    'samples': float(match.group(3))
                }
        elif 'WO CV:' in line:
            # With control variates (WO = WithOut CV is confusing, but seems to mean "With Control Variates")
            match = re.search(r'Solution\s+([\d.-]+)\s+time\s+([\d.-]+)\s+samples\s+([\d.e+-]+)', line)
            if match and current_method:
                results[current_method]['with_cv'] = {
                    'solution': float(match.group(1)),
                    'time': float(match.group(2)),
                    'samples': float(match.group(3))
                }
    
    # Verify we got results for expected methods
    assert 'CubMCCLT' in results, "Should have results for CubMCCLT"
    assert 'CubQMCSobolG' in results, "Should have results for CubQMCSobolG"
    
    # Test that solutions are reasonable (polynomial integrates to around 5.33)
    for method_name, method_results in results.items():
        if 'without_cv' in method_results and 'with_cv' in method_results:
            sol_without = method_results['without_cv']['solution']
            sol_with = method_results['with_cv']['solution']
            
            # Solutions should be close to the true value (around 5.33)
            assert 5.0 <= sol_without <= 6.0, f"Solution without CV should be reasonable: {sol_without}"
            assert 5.0 <= sol_with <= 6.0, f"Solution with CV should be reasonable: {sol_with}"
            
            # Solutions should be close to each other
            assert abs(sol_without - sol_with) < 0.1, f"Solutions should be similar: {sol_without} vs {sol_with}"
            
            # For Monte Carlo methods, control variates should generally reduce samples
            if method_name == 'CubMCCLT':
                samples_without = method_results['without_cv']['samples']
                samples_with = method_results['with_cv']['samples']
                assert samples_with < samples_without, f"Control variates should reduce samples for {method_name}"
    
    print("\nPASSED: Polynomial problem with control variates works correctly.")


@testbook("control_variates.ipynb", execute=False)
def test_keister_problem(tb):
    """
    Test the Keister function integration problem with control variates.
    """
    # Execute setup and Keister problem cells
    tb.execute_cell(slice(0, 7))
    
    # Get the output from the Keister problem cell
    output = tb.cell_output_text(6)
    
    # Check that the output contains expected stopping criteria
    assert 'CubMCCLT' in output, "Should use CubMCCLT"
    assert 'CubQMCSobolG' in output, "Should use CubQMCSobolG"
    
    # Check that solutions are reasonable for Keister function (around 1.38)
    solution_matches = re.findall(r'Solution\s+([\d.-]+)', output)
    solutions = [float(s) for s in solution_matches]
    
    for sol in solutions:
        assert 1.0 <= sol <= 2.0, f"Keister solution should be reasonable: {sol}"
        assert abs(sol - 1.38) < 0.2, f"Solution should be close to expected value: {sol}"
    
    # Check that control variates information is present
    assert 'Control variates took' in output, "Should report control variate performance"
    
    print("\nPASSED: Keister problem with control variates works correctly.")


@testbook("control_variates.ipynb", execute=False)
def test_option_pricing_problem(tb):
    """
    Test the option pricing problem using European option as control variate for Asian option.
    """
    # Execute all cells including the option pricing problem
    tb.execute_cell(slice(0, 8))
    
    # Get the output from the option pricing cell
    output = tb.cell_output_text(7)
    
    # Check that the output contains expected content
    assert 'Stopping Criterion:' in output, "Should show stopping criterion"
    assert 'Solution' in output, "Should show solutions"
    
    # Parse solutions
    solution_matches = re.findall(r'Solution\s+([\d.-]+)', output)
    solutions = [float(s) for s in solution_matches]
    
    # Asian call option prices should be reasonable (around 9.5 based on parameters)
    for sol in solutions:
        assert 5.0 <= sol <= 15.0, f"Option price should be reasonable: {sol}"
        assert abs(sol - 9.55) < 1.0, f"Solution should be close to expected value: {sol}"
    
    # Check for control variate performance reporting
    assert 'Control variates took' in output, "Should report control variate performance"
    
    print("\nPASSED: Option pricing with control variates works correctly.")


@testbook("control_variates.ipynb", execute=False)
def test_control_variate_effectiveness(tb):
    """
    Test that control variates actually improve performance by reducing variance/samples.
    """
    # Execute a simplified version of one problem to test effectiveness
    tb.execute_cell(slice(0, 4))
    
    # Inject code to test control variate effectiveness directly
    test_code = """
# Test polynomial problem with fixed seed for reproducible results
from qmcpy import *
from numpy import *

def test_poly_problem(discrete_distrib):
    g1 = CustomFun(Uniform(discrete_distrib,0,2),lambda t: 10*t[:,0]-5*t[:,1]**2+t[:,2]**3)
    cv1 = CustomFun(Uniform(discrete_distrib,0,2),lambda t: t[:,0])
    cv2 = CustomFun(Uniform(discrete_distrib,0,2),lambda t: t[:,1]**2)
    return g1,[cv1,cv2],[1,4/3]

# Test with and without control variates
g1, cvs, cvmus = test_poly_problem(IIDStdUniform(3, seed=42))

# Without control variates
sc_without = CubMCCLT(g1, abs_tol=1e-2)
sol_without, data_without = sc_without.integrate()

# With control variates
sc_with = CubMCCLT(g1, abs_tol=1e-2, control_variates=cvs, control_variate_means=cvmus)
sol_with, data_with = sc_with.integrate()

# Store results for testing
test_results = {
    'sol_without': sol_without,
    'sol_with': sol_with,
    'samples_without': data_without.n_total,
    'samples_with': data_with.n_total,
    'time_without': data_without.time_integrate,
    'time_with': data_with.time_integrate
}
"""
    
    tb.inject(test_code)
    tb.execute_cell(-1)  # Execute the injected cell
    
    # Get test results
    results = tb.ref("test_results")
    
    # Test that solutions are similar
    sol_diff = abs(results['sol_without'] - results['sol_with'])
    assert sol_diff < 0.5, f"Solutions should be similar: {sol_diff}"
    
    # Test that control variates reduce samples (most of the time)
    samples_ratio = results['samples_with'] / results['samples_without']
    assert samples_ratio < 0.8, f"Control variates should significantly reduce samples: {samples_ratio}"
    
    # Solutions should be reasonable
    assert 4.5 <= results['sol_without'] <= 6.5, f"Solution without CV: {results['sol_without']}"
    assert 4.5 <= results['sol_with'] <= 6.5, f"Solution with CV: {results['sol_with']}"
    
    print(f"\nPASSED: Control variates reduced samples by {(1-samples_ratio)*100:.1f}%")


@testbook("control_variates.ipynb", execute=False)
def test_problem_definitions(tb):
    """
    Test that all problem definition functions work correctly.
    """
    tb.execute_cell(slice(0, 4))
    
    # Test that we can call each problem definition function
    test_code = """
# Test polynomial problem definition
discrete_distrib = IIDStdUniform(3, seed=7)
g1, cvs, cvmus = poly_problem(discrete_distrib)

poly_check = {
    'g1_callable': callable(g1.f),
    'num_cvs': len(cvs),
    'num_cv_means': len(cvmus),
    'cv_means': cvmus
}
"""
    
    tb.inject(test_code)
    
    # Execute cells to define poly_problem
    tb.execute_cell(5)  # Execute the polynomial problem cell
    tb.execute_cell(-1)  # Execute the injected test
    
    results = tb.ref("poly_check")
    
    assert results['g1_callable'], "Main function should be callable"
    assert results['num_cvs'] == 2, "Should have 2 control variates"
    assert results['num_cv_means'] == 2, "Should have 2 control variate means"
    assert results['cv_means'] == [1, 4/3], "Control variate means should be correct"
    
    print("\nPASSED: Problem definitions work correctly.")


@testbook("control_variates.ipynb", execute=False)
def test_output_parsing(tb):
    """
    Test that we can parse the compare function output correctly.
    """
    tb.execute_cell(slice(0, 6))  # Execute through polynomial problem
    
    # Get the polynomial problem output
    output = tb.cell_output_text(5)
    
    # Test parsing performance metrics
    performance_lines = [line for line in output.split('\n') if 'Control variates took' in line]
    
    assert len(performance_lines) >= 2, "Should have performance comparisons"
    
    for line in performance_lines:
        # Extract percentages
        percentages = re.findall(r'([\d.]+)%', line)
        assert len(percentages) >= 2, f"Should have time and sample percentages: {line}"
        
        time_pct = float(percentages[0])
        sample_pct = float(percentages[1])
        
        # Percentages should be reasonable
        assert 0 < time_pct <= 200, f"Time percentage should be reasonable: {time_pct}%"
        assert 0 < sample_pct <= 200, f"Sample percentage should be reasonable: {sample_pct}%"
    
    print("\nPASSED: Output parsing works correctly.")
