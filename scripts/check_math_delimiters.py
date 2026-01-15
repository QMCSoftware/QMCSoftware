#!/usr/bin/env python3
import json
import os

root = "."
problem_files = []

for dirpath, dirnames, filenames in os.walk(root):
    # skip virtualenvs, build, .git
    if any(
        skip in dirpath
        for skip in [".git", "__pycache__", "build", "lib.macosx", "qmcpy.egg-info"]
    ):
        continue
    for fn in filenames:
        if fn.endswith(".md") or fn.endswith(".rst") or fn.endswith(".ipynb"):
            path = os.path.join(dirpath, fn)
            try:
                if fn.endswith(".ipynb"):
                    with open(path, "r", encoding="utf-8") as f:
                        nb = json.load(f)
                    text = ""
                    for cell in nb.get("cells", []):
                        # only markdown/source cells
                        if (
                            cell.get("cell_type") == "markdown"
                            or cell.get("cell_type") == "raw"
                        ):
                            src = cell.get("source", [])
                            if isinstance(src, list):
                                text += "\n".join(src) + "\n"
                            else:
                                text += str(src) + "\n"
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue
            # Remove fenced code blocks (```...```) and inline code (`...`) to avoid counting shell
            # variables like $PATH as math delimiters.
            import re

            # remove fenced code blocks
            text_nocode = re.sub(r"```[\s\S]*?```", "", text)
            # remove inline code spans
            text_nocode = re.sub(r"`[^`]*`", "", text_nocode)

            # remove common currency uses like $10,000 to avoid false positives
            text_nocode = re.sub(r"\$\s*\d[\d,\.]*", "", text_nocode)

            total_dollars = text_nocode.count("$")
            double_dollars = text_nocode.count("$$")
            # single-dollar characters not part of $$ sequences
            single_dollars = total_dollars - 2 * double_dollars
            # if odd number of single dollars, likely an unmatched delimiter
            if single_dollars % 2 == 1:
                problem_files.append(
                    (path, total_dollars, double_dollars, single_dollars)
                )

if not problem_files:
    print("No files with odd number of single $ delimiters found.")
else:
    print("Files with odd single-$ counts (possible unmatched $ delimiters):")
    for p, tot, dbl, single in problem_files:
        print(f"{p}: total $={tot}, $$ pairs={dbl}, single $ count={single}")

# Exit code 0 always
