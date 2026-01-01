#!/usr/bin/env python3
"""
Conservatively convert inline $...$ math to \(...\) in .md files and notebook markdown cells.
Skips code fences and inline code spans. Only replaces $...$ where the interior
contains at least one backslash, brace, caret, underscore, letter, or digit (likely math).
"""
import io
import json
import re
import sys
from pathlib import Path

# pattern: match single-dollar inline math (not $$), with interior containing math-like chars
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(?=[^\n]*[\\{}\^_A-Za-z0-9])([^\n]*?)(?<!\$)\$(?!\$)")

CODE_FENCE_RE = re.compile(r"^(```|~~~)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")


def convert_md_text(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out = []
    in_fence = False
    fence_delim = None
    for line in lines:
        m = CODE_FENCE_RE.match(line)
        if m:
            if not in_fence:
                in_fence = True
                fence_delim = m.group(1)
            else:
                in_fence = False
                fence_delim = None
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        # avoid changing inline code spans
        parts = []
        last = 0
        for im in INLINE_CODE_RE.finditer(line):
            seg = line[last:im.start()]
            seg = INLINE_MATH_RE.sub(r"\\(\1\\)", seg)
            parts.append(seg)
            parts.append(im.group(0))
            last = im.end()
        tail = line[last:]
        tail = INLINE_MATH_RE.sub(r"\\(\1\\)", tail)
        parts.append(tail)
        out.append("".join(parts))
    return "".join(out)


def process_md_file(path: Path) -> bool:
    text = path.read_text(encoding="utf8")
    new = convert_md_text(text)
    if new != text:
        path.write_text(new, encoding="utf8")
        print(f"Patched: {path}")
        return True
    return False


def process_ipynb(path: Path) -> bool:
    j = json.loads(path.read_text(encoding="utf8"))
    changed = False
    for cell in j.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        new = convert_md_text(src)
        if new != src:
            # preserve as list of lines if original was list
            cell["source"] = new.splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(j, indent=1, ensure_ascii=False)+"\n", encoding="utf8")
        print(f"Patched notebook markdown: {path}")
    return changed


def main(paths):
    any_changed = False
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"Not found: {path}")
            continue
        if path.suffix == ".md":
            any_changed |= process_md_file(path)
        elif path.suffix == ".ipynb":
            any_changed |= process_ipynb(path)
        else:
            print(f"Skipping unsupported file: {path}")
    if any_changed:
        print("Done: changes applied.")
    else:
        print("Done: no changes needed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: fix_single_dollar_math.py <file> [files...]")
        sys.exit(2)
    main(sys.argv[1:])
