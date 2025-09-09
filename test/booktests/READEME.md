# Booktests in QMCPy

To execute an individual booktest file, e.g., tb_acm_toms_sorokin_2025.py, run the following command in a terminal:

```{bash}
    cd test/booktests && python -m pytest tb_acm_toms_sorokin_2025.py -v
```

To execute all booktest files sequentially, run the following command in a terminal:

```{bash}
    make booktests_no_docker
```

To execute all booktest files in parallel, run the following command in a terminal:

```{bash}
    make booktests_parallel_no_docker
```

For a demo, see the Jupyter notebook, `demos/parsl_fest_2025.ipynb`.

