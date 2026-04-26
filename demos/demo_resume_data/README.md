# Resume Demo  

This folder contains notebooks, scripts, and saved checkpoint files for the QMCPy resume's demo workflow.

## Contents

- `accuracy_and_resume.ipynb`:   Blog about the feature.

- `resume_examples.ipynb`:   Quick how-to about the feature.

- `check_resume.py`: Runs a cross-method comparison of:
  - a loose run followed by resume with a tighter tolerance
  - a fresh tight run from scratch

  It also captures per-iteration diagnostic logs, writes text reports into
  `output/`, and includes focused multilevel examples for `CubMLMC`,
  `CubMLMCCont`, `CubMLQMC`, and `CubMLQMCCont`.

- `resume_util.py`: Shared helpers for the resume demo scripts, including
  diagnostic capture, case execution, and text-report generation.

- `output/`:  Generated reports and serialized checkpoint examples.
  
## Generated Reports

Running

```bash
python demos/demo_resume_data/check_resume.py
```

writes the following file into `output/`:
- `check_resume_summary.txt`: Combined report of the loose-then-resume and fresh tight-tolerance workflows for each stopping criterion, including captured iteration logs.

The script currently runs with `main(throttle_iterations=False)` when executed directly, so every logged iteration is printed.
