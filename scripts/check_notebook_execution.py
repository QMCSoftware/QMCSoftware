#!/usr/bin/env python3
"""Check that demo notebooks were executed cleanly from a fresh kernel: every
code cell's ``execution_count`` forms the sequence 1, 2, 3, ... with no gaps,
duplicates, or out-of-order numbers, and no cell's outputs contain an error.

A non-sequential execution_count means the saved notebook was NOT run
top-to-bottom in one pass before saving: a cell was added, deleted, or
re-run out of order after an earlier partial run, so the saved outputs may
no longer reflect the saved source. This is cheap to introduce while
iterating (edit one cell, forget to rerun everything below it) and easy to
miss on casual review, since the notebook still looks fine until someone
actually checks the numbers.

Usage:
    python scripts/check_notebook_execution.py [PATH ...] [--strict] [--quiet]
                                                [--diff [REF]]

PATH defaults to every notebook under demos/. ``--diff [REF]`` (REF defaults
to ``develop``) restricts scanning to notebooks that changed relative to
REF -- committed on the branch, modified in the working tree, or untracked.
Informational by default; ``--strict`` turns any finding into a nonzero exit
code, for use in `make check`/CI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ref_style import _changed_files, _display_path, _parse_diff_flag, _summary

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_notebook_file(path):
    """Return a list of (location, category, detail) findings for one notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    findings = []
    expected = 0
    for idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        expected += 1
        actual = cell.get("execution_count")
        if actual != expected:
            findings.append((
                f"cell {idx}", "execution-count-not-sequential",
                f"execution_count is {actual!r}, expected {expected} "
                "(notebook was not run top-to-bottom in one pass before saving)",
            ))
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                ename = output.get("ename", "Error")
                evalue = output.get("evalue", "")
                findings.append((
                    f"cell {idx}", "cell-has-error-output",
                    f"saved output includes {ename}: {evalue[:80]}",
                ))
    return findings


def _iter_notebook_files(root):
    yield from sorted((root / "demos").rglob("*.ipynb"))


def main(argv):
    argv, diff_ref = _parse_diff_flag(list(argv))
    strict = "--strict" in argv
    quiet = "--quiet" in argv
    paths = [a for a in argv if not a.startswith("-")]
    root = REPO_ROOT

    if paths:
        nb_files = []
        for p in map(Path, paths):
            p = p.resolve()
            if p.suffix == ".ipynb":
                nb_files.append(p)
            elif p.is_dir():
                nb_files.extend(sorted(p.rglob("*.ipynb")))
    else:
        nb_files = list(_iter_notebook_files(root))

    if diff_ref is not None:
        try:
            changed = _changed_files(diff_ref)
        except RuntimeError as exc:
            print(f"--diff {diff_ref}: skipped ({exc})", file=sys.stderr)
        else:
            nb_files = [f for f in nb_files if f.resolve() in changed]

    total = 0
    by_cat = {}
    per_file = {}
    for f in nb_files:
        try:
            findings = check_notebook_file(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"{_display_path(f, root)}: skipped ({exc})", file=sys.stderr)
            continue
        if findings:
            per_file[f] = findings
            for _, cat, _ in findings:
                by_cat[cat] = by_cat.get(cat, 0) + 1
                total += 1

    if total and not quiet:
        print()
        for f, findings in per_file.items():
            for loc, cat, detail in findings:
                print(f"  - {_display_path(f, root)}:{loc}: {cat}: {detail}")
    print("  - " + _summary(total, len(nb_files), by_cat, f"{len(nb_files)} file(s) scanned"))

    if total == 0:
        print(f"clean  (0 of {len(nb_files)} files)")
    else:
        prefix = "ERROR" if (strict and total) else "WARNING"
        print(f"{prefix}: {len(per_file)} file(s) with issues  ({len(per_file)} of {len(nb_files)} files)")
    return 1 if (strict and total) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
