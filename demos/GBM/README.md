# Geometric Brownian Motion (GBM) Demonstrations

This directory contains implementations and demonstrations of Geometric Brownian Motion using QMCPy and QuantLib libraries.

## Directories

- **`images/`** - Generated plots and figures saved as PNG files
- **`outputs/`** - Output data files and results
- **`code/`** - Additional code and utilities

## Files

### Jupyter Notebooks

- **`gbm_demo.ipynb`** - Comprehensive demonstration of GBM with interactive visualizations and library comparisons
- **`examples.ipynb`** - Examples of MAE analysis across different replications and configurations

### Python Scripts

- **`__init__.py`** - Package initialization file
- **`config.py`** - Configuration parameters for experiments (debug settings, GBM parameters, sampler configurations)
- **`qmcpy_util.py`** - QMCPy sampler creation and path generation utilities
- **`qmcpy_util_replications.py`** - QMCPy path generation with multiple independent replications
- **`quantlib_util.py`** - QuantLib path generation utilities for IID and Sobol samplers
- **`qmcpy_average_mae.py`** - Compute and plot average MAE for QMCPy samplers over multiple replications
- **`quantlib_average_mae`.py** - Compute and plot average MAE for QuantLib samplers over multiple replications
- **`average_mae.py`** - Compute and plot average MAE for both QMCPy and QuantLib samplers over multiple replications
- **`data_util.py`** - Data processing functions for experimental results and benchmarking
- **`latex_util.py`** - LaTeX table generation and formatting utilities for publication
- **`plot_util.py`** - Visualization functions for GBM paths, distributions, theoretical statistics, error comparisons, performance benchmarks, and parameter sweeps

