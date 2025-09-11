# Notebook Tests in QMCPy

To execute an individual testbook file, e.g., `tb_acm_toms_sorokin_2025.py`, run the following command in a terminal:

```{bash}
    cd test/booktests && python -m pytest tb_acm_toms_sorokin_2025.py -v
```

To execute all testbook files sequentially, run the following command in a terminal:

```{bash}
    make booktests_no_docker
```

To execute all testbook files in parallel, run the following command in a terminal:

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

For a demo, see the Jupyter notebook, `demos/parsl_fest_2025.ipynb`.

