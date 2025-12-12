# Notebook Tests in QMCPy

## Contents

- **`.gitignore`**: ignores temporary files and generated test outputs in this directory.
- **`__init__.py`**: package marker for the `test.booktests` module.
- **`generate_test.py`**: script that generates `tb_*.py` test files from notebooks in `demos/`.
- **`parsl_test_runner.py`**: helper harness used to run Parsl-based notebook tests and coordinate workers.
- **`READEME.md`**: this documentation file describing how to run and manage the notebook tests.

## Overview

To execute an individual testbook file, e.g., `tb_acm_toms_sorokin_2025.py`, run the following command in a terminal:

```{bash}
    cd test/booktests && python -m pytest tb_acm_toms_sorokin_2025.py -v
```

To execute all testbook files sequentially, run the following command in a terminal:

```{bash}
    make booktests_no_docker
```

To execute all testbook files in parallel using Parsl, run the following command in a terminal:

```{bash}
    make booktests_parallel_no_docker
```

To execute say two testbook files sequentially, run the following command in a terminal:

```{bash}
    make booktests_no_docker TESTS="tb_Argonne_2023_Talk_Figures tb_Purdue_Talk_Figures"
```

To execute say two testbook files in parallel, run the following command in a terminal:

```{bash}
    make booktests_parallel_no_docker TESTS="tb_Argonne_2023_Talk_Figures tb_Purdue_Talk_Figures"
```

For a demo, see the Jupyter notebook, `demos/talk_paper_demos/Parslfest_2025/`.

