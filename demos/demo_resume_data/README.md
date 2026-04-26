# Demo Resume Data

This folder contains notebooks, scripts, and saved checkpoint files for the QMCPy resume's demo workflow.

## Contents

- `accuracy_and_resume.ipynb`:   Blog about the feature.

- `resume_examples.ipynb`:   Quick how-to about the feature.

- `check_resume.py`: Runs a cross-method comparison of:
  - a loose run followed by resume with a tighter tolerance
  - a fresh tight run from scratch

  It also captures per-iteration diagnostic logs, writes text reports into
  `output/`, and includes focused multilevel examples for `CubMCML`,
  `CubMCMLCont`, `CubQMCML`, and `CubQMCMLCont`.

- `resume_util.py`: Shared helpers for the resume demo scripts, including
  diagnostic capture, case execution, and text-report generation.

- `output/`:  Generated reports and serialized checkpoint examples.
  
## Generated Reports

Running

```bash
python demos/demo_resume_data/check_resume.py
```

writes the following files into `output/`:

- `check_loose_plus_resume.txt`:  Summary of the loose-then-resume workflow for each stopping criterion, plus the captured loose and resume iteration logs.

- `check_fresh.txt`:  Summary of a fresh tight-tolerance run for each stopping criterion, plus the captured fresh iteration logs.

The script currently runs with `main(throttle_iterations=False)` when executed directly, so every logged iteration is printed.

## Saved Checkpoints

The `.pkl` and `.pkl.gz` files in `output/` are serialized `Data` objects used by the resume demos.
