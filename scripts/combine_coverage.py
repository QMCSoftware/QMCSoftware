#!/usr/bin/env python3
"""Combine multiple coverage data files and generate reports.

Usage:
  python scripts/combine_coverage.py --dir coverage-data --outdir coverage_html

This script finds files named `.coverage*` or `coverage.json` under the
given directory, runs `coverage combine` on them and produces terminal,
xml and HTML reports. It is cross-platform as long as Python and the
`coverage` package are installed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_coverage_files(root: Path):
    files = []
    for p in root.rglob(".coverage*"):
        if p.is_file():
            files.append(str(p.resolve()))
    for p in root.rglob("coverage.json"):
        if p.is_file():
            files.append(str(p.resolve()))
    # Deduplicate while preserving order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def run(cmd, check=True, **kwargs):
    print("+ ", " ".join(cmd))
    return subprocess.run(cmd, check=check, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="coverage-data", help="Directory with downloaded artifacts")
    parser.add_argument("--outdir", default="coverage_html", help="HTML output directory")
    parser.add_argument("--keep", action="store_true", help="Pass --keep to coverage combine")
    args = parser.parse_args(argv)

    root = Path(args.dir)
    if not root.exists():
        print(f"Coverage directory does not exist: {root}")
        sys.exit(1)

    files = find_coverage_files(root)
    if not files:
        print("No coverage files found under", root)
        sys.exit(1)

    # Ensure coverage is installed
    try:
        import coverage  # noqa: F401
    except Exception:
        print("The 'coverage' package is required. Install with: pip install coverage")
        sys.exit(1)

    # Combine using the coverage CLI; pass --keep if requested
    cmd = [sys.executable, "-m", "coverage", "combine"]
    if args.keep:
        cmd.append("--keep")
    cmd.extend(files)
    run(cmd)

    # Reports
    run([sys.executable, "-m", "coverage", "report", "-m"])
    run([sys.executable, "-m", "coverage", "xml", "-o", "coverage.xml"])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "coverage", "html", "-d", str(outdir)])

    print(f"HTML report available at: {outdir / 'index.html'}")


if __name__ == "__main__":
    main()
