# Resume Demo  

This folder contains notebooks and helper scripts for demonstrating QMCPy's
resume workflow.

Current scope: the resume workflow in this demo covers `CubMCCLTVec`,
`CubQMCLatticeG`, `CubQMCNetG`, `CubQMCBayesLatticeG` (`CubBayesLatticeG` alias),
`CubBayesNetG`, `CubQMCRepStudentT`,
`CubMLMC`, `CubMLMCCont`, `CubMLQMC`, and `CubMLQMCCont`. It does not cover
`CubMCCLT`, `CubMCG`, or `PFGPCI`, which currently raise `ParameterError` when
`resume=...` is supplied.

## Contents

- `accuracy_and_resume.ipynb`:   Long-form walkthrough of the feature.

- `resume_examples.ipynb`:   Compact how-to for the feature.

- `check_resume.py`: Runs a cross-method comparison of:
  - a loose run followed by resume with a tighter tolerance
  - a fresh tight run from scratch

  It also stores per-iteration diagnostic logs on the solver/data objects, writes text reports into
  `output/`, and includes focused multilevel examples for `CubMLMC`,
  `CubMLMCCont`, `CubMLQMC`, and `CubMLQMCCont`.

- `resume_util.py`: Shared helpers for the resume demo scripts, including
  case execution, log formatting, and text-report generation.

- `output/`: Generated reports and checkpoint files created when you run the
  notebooks or `check_resume.py`. These artifacts are intentionally not
  tracked in git.
  
## Generated Reports

Running

```bash
python demos/demo_resume_data/check_resume.py
```

creates the following file in `output/`:
- `check_resume_summary.txt`: Combined report of the loose-then-resume and fresh tight-tolerance workflows for each stopping criterion, including stored iteration logs.

Each solver now stores an in-memory iteration log after `integrate()`. Use `history_df` or `get_iteration_log()` for a formatted table, or `format_iteration_log()` / `print_iteration_log()` for text replay. By default that stored history is throttled; enabling `trace_iterations` lets you opt into live printing, and `verbose=True` stores every iteration. On resumed runs, the new `RESUME` stage is appended to the saved loose-stage history. The demo helper `print_stage_summary(...)` can now rebuild the Loose/Resumed/Fresh summary directly from those stored histories.
