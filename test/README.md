# Test Targets Guide

This document describes the available test targets in the Makefile for QMCSoftware. All targets use pytest with parallel execution (via `pytest-xdist`) when available.

## Quick Reference

| Target | Purpose | Speed | Use Case |
|--------|---------|-------|----------|
| `make tests_fast` ⚡ | **Recommended**: All tests in parallel | Fast | Default choice: doctests + unittests + booktests concurrently |
| `make tests_no_docker` | All tests sequentially (no Docker) | Slow | Conservative validation; isolates flaky tests |
| `make tests` | All tests with Docker support | Very Slow | Complete validation with GPU/Docker dependencies |
| `make unittests` | Unit tests only | Fast | Quick feedback on code changes |
| `make doctests_no_docker` | Doctests (excludes GPU/Docker) | Moderate | Validate docstring examples |
| `make doctests` | All doctests with Docker | Slow | Full docstring validation |
| `make booktests_no_docker` | Jupyter notebook tests | Slow | Validate demo notebooks |
| `make booktests_parallel_no_docker` | Notebook tests with Parsl parallelization | Variable | Distributed notebook execution |
| `make coverage` | Display coverage report | Instant | View test coverage summary |
| `make delcoverage` | Reset coverage tracking | Instant | Start fresh coverage analysis |


## Detailed Descriptions

### Core Test Targets (Recommended)

#### `make tests_fast` ⚡ **RECOMMENDED**
**Fastest option**: Runs doctests, unittests, and booktests **concurrently** in background processes.
- **Parallelization**: All three test families run simultaneously, not sequentially
- **Cleanup**: Removes invalid distribution artifacts before running
- **Time**: ~30–60 seconds (depending on CPU cores and notebook complexity)
- **Coverage**: Full summary report at the end
- **Use when**: You want comprehensive testing with maximum speed

#### `make tests_no_docker`
Runs all tests sequentially: doctests, unittests, and generates coverage reports (excludes Docker-dependent tests).
- **Cleans** invalid distribution artifacts before running (via `scripts/cleanup_invalid_dist.py`)
- **Sequence**: doctests_no_docker → unittests → coverage report
- **Time**: ~60–120 seconds
- **Coverage**: Full summary report
- **Use when**: Pre-commit or CI/CD validation (without Docker)

#### `make tests`
Runs all tests **sequentially** with full Docker support (for GPU-heavy and UMBridge tests).
- **Includes**: Full doctests suite (with umbridge and markdown validation)
- **Time**: ~120–180+ seconds
- **Coverage**: Full summary report
- **Use when**: Complete validation with Docker dependencies available

---

### Doctest Targets

#### `make doctests_no_docker` (Composite)
Runs doctests excluding GPU and Docker dependencies.
- **Composition**: `doctests_minimal` + `doctests_torch` + `doctests_gpytorch` + `doctests_botorch`
- **Time**: ~15–30 seconds
- **Use when**: Validating docstring examples

#### `make doctests` (Composite)
Runs all doctests including Docker-dependent UMBridge and markdown validation.
- **Composition**: `doctests_markdown` + `doctests_minimal` + `doctests_torch` + `doctests_gpytorch` + `doctests_botorch` + `doctests_umbridge`
- **Time**: Variable (Docker startup overhead)
- **Use when**: Full docstring validation with Docker available

#### `make doctests_minimal` (Building Block)
Core doctests excluding all optional dependencies (PyTorch, GPyTorch, BoTorch, UMBridge).
- **Modules tested**: Main qmcpy modules
- **Time**: ~5–10 seconds
- **Note**: Usually called via `doctests_no_docker` or `doctests`; rarely used standalone

#### `make doctests_torch` (Building Block)
Doctests for PyTorch-dependent modules.
- **Modules tested**: `qmcpy/fast_transform/ft_pytorch.py`, `qmcpy/kernel/*.py`, `qmcpy/util/*shift*.py`
- **Time**: ~3–5 seconds
- **Note**: Usually called via `doctests_no_docker`; rarely used standalone

#### `make doctests_gpytorch` (Building Block)
Doctests for GPyTorch integration.
- **Modules tested**: `qmcpy/stopping_criterion/pf_gp_ci.py`
- **Time**: ~3–5 seconds
- **Note**: Usually called via `doctests_no_docker`; rarely used standalone

#### `make doctests_botorch` (Building Block)
Doctests for BoTorch integration.
- **Modules tested**: `qmcpy/integrand/hartmann6d.py`
- **Time**: ~2–3 seconds
- **Note**: Usually called via `doctests_no_docker`; rarely used standalone

#### `make doctests_umbridge`
Doctests for UMBridge wrapper (requires Docker).
- **Modules tested**: `qmcpy/integrand/umbridge_wrapper.py`
- **Time**: ~10–15 seconds (includes Docker startup)
- **Dependencies**: Docker must be running
- **Note**: Usually called via `doctests`; rarely used standalone

#### `make doctests_markdown`
Validates embedded Python code in markdown files under `docs/`.
- **Tools**: Uses `phmutest` (markdown test utility)
- **Time**: ~3–5 seconds
- **Note**: Usually called via `doctests`; rarely used standalone

---

### Unit & Notebook Test Targets

#### `make unittests`
Runs unit tests from the `test/` directory using pytest with parallel workers (when `pytest-xdist` is installed).
- **Time**: ~13–30 seconds (depending on CPU cores)
- **Coverage**: Incremental coverage report appended to `.coverage`
- **Use when**: Quick feedback on code changes

#### `make booktests_no_docker`
Generates and runs tests from Jupyter notebooks in the `demos/` folder using unittest discovery.
- **Automatically generates** missing test files in `test/booktests/`
- **Cleans** cache and temporary files before running
- **Time**: Highly variable (5 seconds to 5+ minutes depending on notebook complexity)
- **Coverage**: Incremental
- **Use when**: Validating demo notebooks or documentation

#### `make booktests_parallel_no_docker`
Runs notebook tests with **Parsl distributed parallelization** for compute-heavy demos.
- **Parallelization**: Uses Parsl framework for task scheduling
- **Cleanup**: Removes temporary outputs (EPS, JPG, PDF, PNG files, logs, etc.)
- **Time**: Highly variable (depends on Parsl workers and notebook complexity)
- **Optional parameters**: `TESTS="tb_notebook1 tb_notebook2"` to run specific tests
- **Dependencies**: Parsl must be installed and configured
- **Use when**: Running large notebook suites with distributed compute resources

#### `make tests_parallel_no_docker`
Runs only unit tests with parallel pytest workers (no doctests or booktests).
- **Time**: ~13–20 seconds
- **Coverage**: Incremental
- **Use when**: Testing unit tests only in parallel mode

---

### Helper / Internal Targets

#### `make check_booktests`
Validates that all Jupyter notebooks in `demos/` have corresponding test files in `test/booktests/`.
- **Output**: Lists missing test files and notebook/test file counts
- **Note**: Called automatically by `booktests_no_docker`; rarely used standalone

#### `make generate_booktests`
Auto-generates missing test stub files for notebooks.
- **Output**: Reports any generated files
- **Note**: Called automatically by `booktests_no_docker`; rarely used standalone

#### `make coverage`
Displays the current coverage report (must run other targets first to accumulate coverage data).
- **Output**: Terminal summary of coverage percentages per file/module
- **Note**: Coverage data is appended from previous runs; use `make delcoverage` to reset

#### `make delcoverage`
Deletes `.coverage` and `coverage.json` files to reset coverage tracking.
- **Use before**: Running a fresh coverage report without accumulated data

## Redundancy Analysis & Status

### Removed Redundant Target ✅

#### `make tests_parallel_no_docker` (REMOVED)
- **Was redundant**: Ran only unit tests in parallel. `make tests_fast` is a strict superset (doctests + unittests + booktests in parallel).
- **Status**: **Removed from Makefile** to simplify maintenance and reduce user confusion.
- **Migration**: Users should use `make tests_fast` instead (faster, more comprehensive).

---

## Currently Active Targets: Justification

---

### Target Dependency Graph

```
tests_fast ⚡ (RECOMMENDED)
├── doctests_no_docker
│   ├── doctests_minimal
│   ├── doctests_torch
│   ├── doctests_gpytorch
│   └── doctests_botorch
├── unittests
└── booktests_no_docker
    ├── check_booktests
    ├── generate_booktests
    └── clean_local_only_files

tests_no_docker
├── doctests_no_docker
├── unittests
└── coverage

tests (full with Docker)
├── doctests
│   ├── doctests_markdown
│   ├── doctests_minimal
│   ├── doctests_torch
│   ├── doctests_gpytorch
│   ├── doctests_botorch
│   └── doctests_umbridge
├── unittests
└── coverage

booktests_parallel_no_docker
├── check_booktests
├── generate_booktests
├── clean_local_only_files
└── [Parsl distributed execution]
```

### Automatic Parallel Execution
- If `pytest-xdist` is installed, tests run with `-n auto` (detected by `scripts/pytest_xdist.py`)
- Override: `make PYTEST_XDIST="-n 4" unittests` to force 4 workers

### Selective Notebook Tests
Run specific notebook tests:
```bash
make booktests_no_docker TESTS="tb_quickstart tb_pricing_options"
```

### Environment Cleanup
The test targets automatically call `scripts/cleanup_invalid_dist.py --apply` to remove corrupted distribution artifacts (e.g., invalid seaborn entries).

## Reproducibility

All tests use deterministic seeds (e.g., `seed=42`, `seed=7`) where applicable to ensure reproducible results across runs.

## Workflow Recommendations

### For Development
```bash
make unittests              # Quick feedback (13–30s)
make doctests_no_docker     # Validate docstrings (15–30s)
```

### Before Committing
```bash
make tests_fast             # Comprehensive, fast (30–60s)
```

### For CI/CD or Full Validation
```bash
make tests_no_docker        # Sequential, safe (60–120s)
```

## Troubleshooting

**Issue**: Tests fail with "command not found" or pytest not recognized
- **Solution**: Ensure the conda environment is activated: `conda activate qmcpy`

**Issue**: Tests are slow or not parallelized
- **Solution**: Install `pytest-xdist`: `pip install pytest-xdist`

**Issue**: Warnings about invalid distributions
- **Solution**: Cleanup runs automatically; if needed manually: `python scripts/cleanup_invalid_dist.py --apply`

**Issue**: Coverage numbers seem low or cumulative
- **Solution**: Reset coverage with `make delcoverage`, then run tests

---

## Coverage Report Strategy

### Overview
QMCSoftware uses a **multi-platform unified coverage report** approach in GitHub Actions CI. Coverage data from all test types (doctests, unittests, booktests) running on all platforms (Ubuntu, macOS, Windows) is combined into a single comprehensive report.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Matrix Job: tests (windows-latest, macos-latest, ubuntu-latest)  │
├─────────────────────────────────────────────────────────┤
│  1. Clean old coverage files (.coverage*, coverage.json)│
│  2. Run doctests (with --cov-append)                    │
│  3. Run unittests (with --cov-append)                   │
│  4. Run booktests (with --cov-append)                   │
│  5. Upload .coverage & coverage.json as artifacts       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Final Job: combine-coverage (ubuntu-latest)            │
├─────────────────────────────────────────────────────────┤
│  1. Download all coverage artifacts (3 OS runners)      │
│  2. Combine with: coverage combine                      │
│  3. Generate reports:                                   │
│     - coverage report -m (terminal)                     │
│     - coverage xml -o coverage.xml                      │
│     - coverage html -d coverage_html                    │
│  4. Upload final HTML & XML as artifacts                │
└─────────────────────────────────────────────────────────┘
```

### Key Syntax & Configuration

#### 1. Coverage Configuration (`.coveragerc`)
Cross-platform coverage combining requires relative paths to handle different OS path formats (Windows `C:\`, macOS `/Users/`, Ubuntu `/home/`):

**`.coveragerc` settings:**
- `relative_files = True` – Store paths relative to project root
- `[paths]` section maps all OS path variants to common source location
- Enables successful `coverage combine` across Windows, macOS, and Ubuntu runners

#### 2. Makefile Test Targets (Coverage Append Mode)
All test targets use `--cov-append` (pytest) or `coverage run --append` to accumulate coverage within each OS runner:

**Doctest targets:** `doctests_minimal`, `doctests_torch`, `doctests_gpytorch`, `doctests_botorch`, `doctests_umbridge` – all use `--cov-append`

**Unit test target:** `unittests` – uses `--cov-append`

**Notebook test targets:**
- `booktests_no_docker` – uses `coverage run --append`
- `booktests_parallel_no_docker` – Parsl runner internally uses `coverage run --append`
- `booktests_parallel_pytest` – uses `--cov-append`

**Key flags:**
- `--cov qmcpy/` – Target package for coverage measurement
- `--cov-append` – Append to existing `.coverage` data (pytest-cov)
- `coverage run --append` – Append mode for unittest-based tests
- `--cov-report term` – Terminal output after each test run
- `--cov-report json` – Generate `coverage.json` for tracking

#### 3. GitHub Actions Workflow

- **Clean coverage at start of each matrix job:**
- **Upload coverage artifacts after all tests:**
- **Combine in separate job:**

### Local Coverage Workflow

**Run tests and view coverage locally:**

- Clean old coverage
- Run tests (accumulates coverage with --cov-append)
- View combined report


### Benefits

1. **Unified cross-platform coverage** – One report combines Ubuntu, macOS, and Windows execution paths
2. **Comprehensive test coverage** – Includes doctests, unittests, and notebook tests
3. **Artifact persistence** – HTML and XML reports available for download/review in GitHub Actions
4. **Incremental local testing** – `--cov-append` allows building coverage across multiple test runs
5. **CI/CD integration ready** – XML output compatible with Codecov, Coveralls, etc.

### Troubleshooting

**Coverage numbers seem wrong or incomplete:**
- Run `make delcoverage` to clean old data before starting fresh
- Ensure all test commands use `--cov-append` or `coverage run --append`

**GitHub Actions shows no coverage report:**
- Check that `combine-coverage` job downloaded artifacts from all 3 OS runners
- Verify each matrix job successfully uploaded coverage artifacts

**Coverage combining fails locally:**
- Ensure `coverage` package is installed: `pip install coverage`
- Check that `.coverage` files exist before running `coverage combine`

---

## See Also

- [../Makefile](../Makefile) – Full test target definitions
- [../.github/workflows/alltests.yml](../.github/workflows/alltests.yml) – CI coverage workflow
- [../scripts/cleanup_invalid_dist.py](../scripts/cleanup_invalid_dist.py) – Artifact cleanup utility
- [../scripts/pytest_xdist.py](../scripts/pytest_xdist.py) – Parallel execution detection helper


