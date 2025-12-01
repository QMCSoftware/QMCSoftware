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



