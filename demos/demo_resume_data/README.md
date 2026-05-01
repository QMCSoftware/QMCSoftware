# Resume Demo  

This folder contains notebooks and helper scripts for demonstrating QMCPy's
resume workflow.

Current scope: the resume workflow in this demo covers `CubMCCLTVec`,
`CubQMCLatticeG`, `CubQMCNetG`, `CubQMCBayesLatticeG` (`CubBayesLatticeG` alias),
`CubBayesNetG`,
`CubMLMC`, `CubMLMCCont`, `CubMLQMC`, and `CubMLQMCCont`. It does not cover
`CubMCCLT`, `CubMCG`, `CubQMCRepStudentT`, or `PFGPCI`, which currently raise `ParameterError` when
`resume=...` is supplied.

## Contents

- `accuracy_and_resume.ipynb`:   Long-form walkthrough of the feature.

- `resume_examples.ipynb`:   Compact how-to for the feature.

- `check_resume.py`: Runs a cross-method comparison of:
  - a loose run followed by resume with a tighter tolerance
  - a fresh tight run from scratch

  It also captures per-iteration diagnostic logs, writes text reports into
  `output/`, and includes focused multilevel examples for `CubMLMC`,
  `CubMLMCCont`, `CubMLQMC`, and `CubMLQMCCont`.

- `resume_util.py`: Shared helpers for the resume demo scripts, including
  diagnostic capture, case execution, and text-report generation.

- `output/`: Generated reports and checkpoint files created when you run the
  notebooks or `check_resume.py`. These artifacts are intentionally not
  tracked in git.
  
## Generated Reports

Running

```bash
python demos/demo_resume_data/check_resume.py
```

creates the following file in `output/`:
- `check_resume_summary.txt`: Combined report of the loose-then-resume and fresh tight-tolerance workflows for each stopping criterion, including captured iteration logs.

The script currently runs with `main(verbose=True)` when executed directly, so every logged iteration is printed.
