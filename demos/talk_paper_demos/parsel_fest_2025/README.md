# ParslFest 2025: Accelerating QMCpy Notebook Tests with Parsl

This directory contains demonstration notebooks for the ParslFest 2025 presentation on using Parsl to parallelize QMCpy notebook tests.

**Presentation**: [Accelerating QMCpy Notebook Tests with Parsl](https://www.figma.com/slides/k7EUosssNluMihkYTLuh1F/Parsl-Testbook-Speedup?node-id=1-37&t=WnKcu2QYO8JXvtpP-0)

## Files

- **`01_sequential.ipynb`** - Runs all QMCpy notebook tests sequentially without parallelization to establish a baseline execution time
- **`02_parallel.ipynb`** - Runs the same notebook tests in parallel using Parsl with configurable worker count to measure speedup
- **`03_visualize_speedup.ipynb`** - Visualizes and analyzes the speedup achieved by parallel execution across different worker configurations
- **`.gitignore`** - Git ignore rules for output files and temporary data
- **`output/`** - Directory containing execution timing results and speedup metrics (CSV files)
- **`runinfo/`** - Directory containing Parsl execution logs and runtime information

## Usage

**Important**: Run the notebooks in order (01 → 02 → 03).

### 1. Sequential Baseline
Run `01_sequential.ipynb` first to establish the sequential execution time baseline. This will generate `output/sequential_time.csv`.

### 2. Parallel Execution
Run `02_parallel.ipynb` to execute tests in parallel with Parsl. 

**To test different worker counts:**
- Modify `config.max_workers = 2` (line 65) to different values: 2, 4, 8, etc.
- Re-run the notebook for each worker count to generate separate timing files
- Each run creates `output/parallel_times_{N}.csv` where N is the number of workers

### 3. Visualization
Run `03_visualize_speedup.ipynb` to:


## Load Balancing with LPT Scheduling

The parallel test runner uses the **Longest Processing Time (LPT)** algorithm (also known as Longest Job First) to optimize load balancing across Parsl workers.

### How It Works

1. **Runtime Estimates**: `test_runtimes.py` stores measured execution times for each notebook test from sequential runs.

2. **Optimal Bin Packing**: Before submitting jobs, tests are sorted by estimated runtime (longest first) and assigned to the least-loaded worker using a min-heap.

3. **Balanced Workloads**: This ensures all workers finish at approximately the same time, maximizing parallelism.

### Scalability Analysis

Based on measured runtimes (total sequential time: ~759s):

| Workers | Estimated Makespan | Speedup | Efficiency |
|---------|-------------------|---------|------------|
| 1 | 759s | 1.0x | 100% |
| 2 | 379s | 2.0x | 100% |
| 4 | 190s | 4.0x | 100% |
| 6 | 138s | 5.5x | 92% |
| 8 | 138s | 5.5x | 69% |

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
