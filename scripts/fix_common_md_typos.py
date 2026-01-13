#!/usr/bin/env python3
import os, re

root = "."
replacements = {
    r"\bteh\b": "the",
    r"\bseperate\b": "separate",
    r"\brecieve\b": "receive",
    r"\boccured\b": "occurred",
    r"\bparings\b": "pairings",
    r"\bintial\b": "initial",
    r"\butilites\b": "utilities",
}

changed_files = []
for dirpath, dirnames, filenames in os.walk(root):
    if any(
        skip in dirpath
        for skip in [".git", "__pycache__", "build", "lib.macosx", "qmcpy.egg-info"]
    ):
        continue
    for fn in filenames:
        if not fn.endswith(".md"):
            continue
        path = os.path.join(dirpath, fn)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # split by fenced code blocks ``` to avoid changing code
        parts = re.split(r"(```[\s\S]*?```)", text)
        newparts = []
        changed = False
        for i, part in enumerate(parts):
            if i % 2 == 1:
                # inside fenced block, leave unchanged
                newparts.append(part)
            else:
                # avoid inline code `...` by splitting
                subparts = re.split(r"(`[^`]*`)", part)
                newsub = []
                for j, sp in enumerate(subparts):
                    if j % 2 == 1:
                        newsub.append(sp)
                    else:
                        s = sp
                        for pat, repl in replacements.items():
                            s, replcount = re.subn(pat, repl, s)
                            if replcount > 0:
                                changed = True
                        newsub.append(s)
                newparts.append("".join(newsub))
        newtext = "".join(newparts)
        if changed and newtext != text:
            # backup original
            with open(path + ".bak", "w", encoding="utf-8") as f:
                f.write(text)
            with open(path, "w", encoding="utf-8") as f:
                f.write(newtext)
            changed_files.append(path)

if changed_files:
    print("Updated files:")
    for p in changed_files:
        print(" -", p)
else:
    print("No conservative typos found to fix.")
