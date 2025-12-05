# ParslFest 2025: Accelerating QMCPy Notebook Tests with Parsl

This directory contains demonstration notebooks for the ParslFest 2025 presentation on using Parsl to parallelize QMCPy notebook tests.

**Presentation**: [Accelerating QMCPy Notebook Tests with Parsl](https://www.figma.com/slides/k7EUosssNluMihkYTLuh1F/Parsl-Testbook-Speedup?node-id=1-37&t=WnKcu2QYO8JXvtpP-0)

## Files

- **`01_sequential.ipynb`** - Runs QMCPy notebook tests sequentially without parallelization to establish a baseline execution time
- **`02_parallel.ipynb`** - Runs the same notebook tests in parallel using Parsl with configurable worker count to measure speedup
- **`02_parallel_mac.ipynb`** - macOS-specific version using ThreadPoolExecutor for compatibility
- **`03_visualize_speedup.ipynb`** - Visualizes and analyzes the speedup achieved by parallel execution across different worker configurations
- **`Makefile`** - Automates execution of all notebooks sequentially with multiple worker configurations
- **`.gitignore`** - Git ignore rules for output files and temporary data
- **`output/`** - Directory containing execution timing results, speedup metrics (CSV files), and plots (PNG files)
- **`runinfo/`** - Directory containing Parsl execution logs and runtime information
- **`ParslFest_2025.deck`** -  presentation deck 
- **`ParslFest_2025.pdf`** - Exported PDF of the presentation slides

## Quick Start

### Automated Execution (Recommended)

Run all notebooks automatically with different worker configurations:

```bash
make clean && make all
```

This will:
1. Clean previous outputs
2. Run `01_sequential.ipynb` to establish baseline
3. Run `02_parallel.ipynb` with 2, 4, 8, and 16 workers using `PARSL_MAX_WORKERS` environment variable
4. Run `03_visualize_speedup.ipynb` to generate plots and analysis

**Individual targets:**
```bash
make sequential   # Run only sequential baseline
make parallel     # Run parallel tests with 2,4,8,16 workers
make visualize    # Generate plots from existing data
make clean        # Remove all output files
```

### Manual Execution

**Important**: Run the notebooks in order (01 → 02 → 03).

#### 1. Sequential Baseline
Run `01_sequential.ipynb` to establish the sequential execution time baseline. This generates:
- `output/sequential_time.csv` - Total execution time
- `output/sequential_output_time.csv` - Individual test times
- `output/sequential_output_memory.csv` - Memory usage per test

#### 2. Parallel Execution
Run `02_parallel.ipynb` to execute tests in parallel with Parsl. 

**Environment variable control:**
```python
# Set PARSL_MAX_WORKERS before running:
import os
os.environ['PARSL_MAX_WORKERS'] = '4'  # Use 4 workers
```

Each run creates `output/parallel_times_{N}.csv` and `output/parallel_output_{N}.txt`.

#### 3. Visualization
Run `03_visualize_speedup.ipynb` to generate plots comparing sequential vs parallel execution times and compute speedup ratios.


**Maximum theoretical speedup: 5.5x** (limited by the longest single test: `tb_iris` at 138s).

### Assumptions

1. **Reproducible runtimes**: Test execution times are consistent across runs.
2. **Independent tests**: Each notebook test can run independently without shared state.
3. **Single-threaded tests**: Individual tests do not benefit from multiple cores internally.
4. **No I/O contention**: Parallel tests do not compete for disk or network resources.
5. **Accurate estimates**: Runtime estimates in `test_runtimes.py` reflect actual execution times.

### Updating Runtime Estimates

After running sequential tests, update `test_runtimes.py` with new timing data:

```bash
make booktests_no_docker  # Run sequential tests, note the timing output
# Then update TEST_RUNTIMES dict in test_runtimes.py
```
