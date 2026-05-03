# Resume Demo  

This folder contains notebooks and helper scripts for demonstrating QMCPy's resume workflow.

**Current scope:** the resume workflow in this demo covers `CubMCCLTVec`, `CubQMCLatticeG`, `CubQMCNetG`, `CubQMCBayesLatticeG` (`CubBayesLatticeG` alias), `CubBayesNetG`, `CubQMCRepStudentT`, `CubMLMC`, `CubMLMCCont`, `CubMLQMC`, and `CubMLQMCCont`. It does not cover `CubMCCLT`, `CubMCG`, or `PFGPCI`, which currently raise `ParameterError` when `resume=...` is supplied.

## Contents

- `accuracy_and_resume.ipynb`:   Long-form walkthrough of the feature.

- `resume_examples.ipynb`:   Compact how-to for the feature.

- `check_resume.py`: Runs a cross-method comparison of:
  - a loose run followed by resume with a tighter tolerance
  - a fresh tight run from scratch

- `check_resume_long.py`: Extended version of `check_resume.py` with a more thorough cross-method comparison of:
  - a loose run followed by resume with a tighter tolerance
  - a fresh tight run from scratch

  It also stores per-iteration diagnostic logs on the solver/data objects, writes text reports into `output/`, and includes focused multilevel examples for `CubMLMC`, `CubMLMCCont`, `CubMLQMC`, and `CubMLQMCCont`.

- `resume_util.py`: Shared helpers for the resume demo scripts, including case execution, log formatting, and text-report generation.

- `output/`: Generated reports and checkpoint files created when you run the notebooks or `check_resume.py`. These artifacts are intentionally not tracked in git.
  
## Generated Reports

Running

```bash
python demos/demo_resume_data/check_resume.py
```

creates the following file in `output/`:
- `check_resume_summary.txt`: Combined report of the loose-then-resume and fresh tight-tolerance workflows for each stopping criterion, including stored iteration logs.

Running

```bash
python demos/demo_resume_data/check_resume_long.py
```

creates the following files in `output/`:
- `check_resume_long_summary.txt`: Extended report of the loose-then-resume and fresh tight-tolerance workflows for each stopping criterion, including stored iteration logs and multilevel examples.

## Iteration Logs

Supported solvers/data objects now keep an in-memory iteration history after `integrate()`. Use `get_iteration_log()` to inspect a table view. Stored history is throttled by default; set `trace_iterations=True` for live per-iteration output, and use `verbose=True` to retain every iteration. 
