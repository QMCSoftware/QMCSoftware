#!/usr/bin/env python3
"""Flatten public absolute QMCPy imports to use the top-level package."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Iterator
import json
import os
from pathlib import Path
import re
import sys


SUPPORTED_SUFFIXES = {".ipynb", ".md", ".py", ".pyi", ".rst", ".txt"}
SKIPPED_DIRECTORIES = {
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pdm-build",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "site",
    "venv",
}

# Matching bytes avoids changing line endings or reformatting notebook JSON.
QMCPY_IMPORT_RE = re.compile(
    rb"\bfrom(?P<after_from>[ \t]+)qmcpy"
    rb"(?P<module_path>(?:\.[A-Za-z_][A-Za-z0-9_]*)+)"
    rb"(?P<before_import>[ \t]+)import\b"
    rb"(?P<imported>[ \t]*(?:\([^)]*\)|[^\r\n]*))"
)
PRIVATE_NAME_RE = re.compile(rb"(?<![A-Za-z0-9_])_[A-Za-z0-9_]+")
TEXT_STAR_IMPORT_RE = re.compile(
    rb"^(?P<indent>[ \t]*)from[ \t]+qmcpy[ \t]+import[ \t]+\*[ \t]*"
    rb"(?:\r\n|\n|\r)?$"
)
NOTEBOOK_STAR_IMPORT_RE = re.compile(
    rb'^(?P<json_indent>[ \t]*)"(?P<code_indent>(?:[ \t]|\\t)*)'
    rb"from[ \t]+qmcpy[ \t]+import[ \t]+\*(?:\\r)?(?:\\n)?\""
    rb",?[ \t]*(?:\r\n|\n|\r)?$"
)
STAR_IMPORT_LITERAL = b"from qmcpy import *"
# Bare (already top-level) single-line "from qmcpy import ..." statements, so
# stale comma spacing can be cleaned up even when there's no module path to
# flatten. Anchored to line start, so notebook JSON lines (which have a
# leading quote before "from") never match.
NAMED_IMPORT_LINE_RE = re.compile(
    rb"^(?P<prefix>[ \t]*from[ \t]+qmcpy[ \t]+import)(?P<imported>[ \t][^\r\n]*?)"
    rb"[ \t]*(?:\r\n|\n|\r)?$"
)
COMMA_SPACING_RE = re.compile(rb",(?=\S)")

# A line only looks like an IPython magic/shell escape when '%'/'%%' is
# immediately followed by a name (e.g. "%matplotlib"); "% (x, y)" is a
# modulo-operator continuation and must be left alone.
MAGIC_LINE_RE = re.compile(r"^[ \t]*(?:%{1,2}[A-Za-z_]|!|\?)")


def _star_import_context(line: bytes) -> tuple[bytes, bytes] | None:
    text_match = TEXT_STAR_IMPORT_RE.match(line)
    if text_match:
        return b"text", text_match.group("indent")

    notebook_match = NOTEBOOK_STAR_IMPORT_RE.match(line)
    if notebook_match:
        indentation = (
            notebook_match.group("json_indent")
            + b'"'
            + notebook_match.group("code_indent")
        )
        return b"notebook", indentation

    return None


def _deduplicate_adjacent_star_imports(content: bytes) -> tuple[bytes, int]:
    """Keep the last line in each run of same-scope QMCPy star imports."""

    output: list[bytes] = []
    previous_context: tuple[bytes, bytes] | None = None
    removed_count = 0

    for line in content.splitlines(keepends=True):
        context = _star_import_context(line)
        if context is not None and context == previous_context:
            # Keeping the last line preserves JSON comma and newline placement.
            output[-1] = line
            removed_count += 1
        else:
            output.append(line)
        previous_context = context

    return b"".join(output), removed_count


def _load_qmcpy_public_names(repository_root: Path) -> frozenset[str] | None:
    """Return qmcpy's public top-level names, or None if qmcpy can't be imported."""

    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    try:
        import qmcpy
    except ImportError:
        return None
    return frozenset(name for name in dir(qmcpy) if not name.startswith("_"))


def _names_needing_import(source: str, public_names: frozenset[str]) -> set[str] | None:
    """Return public qmcpy names referenced but not otherwise bound in `source`.

    Returns None if `source` isn't parseable Python. The scan is file-wide
    rather than scope-aware, so a name bound anywhere (even in an unrelated
    scope) is treated as locally defined and excluded, which is the safe
    direction to err in.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    loaded: set[str] = set()
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (loaded if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.alias):
            bound.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)

    return (loaded & public_names) - bound


def _line_ending(line: bytes) -> bytes:
    for ending in (b"\r\n", b"\n", b"\r"):
        if line.endswith(ending):
            return ending
    return b""


def _format_expanded_import(indent: bytes, names: Iterable[str], ending: bytes) -> bytes:
    sorted_names = sorted(names)
    indent_text = indent.decode()
    single_line = f"{indent_text}from qmcpy import {', '.join(sorted_names)}"
    if len(single_line) <= 88:
        return single_line.encode() + ending

    body_lines = [f"{indent_text}from qmcpy import ("]
    body_lines.extend(f"{indent_text}    {name}," for name in sorted_names)
    body_lines.append(f"{indent_text})")
    return ending.join(line.encode() for line in body_lines) + ending


def _expand_text_star_imports(content: bytes, public_names: frozenset[str]) -> tuple[bytes, int]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content, 0

    needed = _names_needing_import(text, public_names)
    if not needed:
        return content, 0

    output: list[bytes] = []
    change_count = 0
    for line in content.splitlines(keepends=True):
        match = TEXT_STAR_IMPORT_RE.match(line)
        if match is None:
            output.append(line)
            continue
        output.append(
            _format_expanded_import(match.group("indent"), needed, _line_ending(line))
        )
        change_count += 1

    return b"".join(output), change_count


def _sanitize_magic_lines(source: str) -> str:
    """Blank out IPython magic/shell-escape lines so the cell can be parsed as Python."""

    return "".join(
        "pass\n" if MAGIC_LINE_RE.match(line) else line
        for line in source.splitlines(keepends=True)
    )


def _notebook_python_source(content: bytes) -> str | None:
    """Concatenate a notebook's code cells into one pseudo-module for analysis."""

    try:
        notebook = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(notebook, dict):
        return None
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return None

    sources = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if isinstance(source, str) and source:
            sources.append(_sanitize_magic_lines(source))

    return "\n\n".join(sources)


def _expand_notebook_star_imports(content: bytes, public_names: frozenset[str]) -> tuple[bytes, int]:
    combined_source = _notebook_python_source(content)
    if combined_source is None:
        return content, 0

    needed = _names_needing_import(combined_source, public_names)
    if not needed:
        return content, 0

    import_text = ("from qmcpy import " + ", ".join(sorted(needed))).encode()

    output: list[bytes] = []
    change_count = 0
    for line in content.splitlines(keepends=True):
        if NOTEBOOK_STAR_IMPORT_RE.match(line) is None:
            output.append(line)
            continue
        output.append(line.replace(STAR_IMPORT_LITERAL, import_text, 1))
        change_count += 1

    return b"".join(output), change_count


def _normalize_comma_spacing(content: bytes) -> tuple[bytes, int]:
    """Ensure a space follows each comma in single-line qmcpy import statements."""

    output: list[bytes] = []
    change_count = 0
    for line in content.splitlines(keepends=True):
        match = NAMED_IMPORT_LINE_RE.match(line)
        if match is None:
            output.append(line)
            continue

        imported = match.group("imported")
        normalized = COMMA_SPACING_RE.sub(b", ", imported)
        if normalized == imported:
            output.append(line)
            continue

        change_count += 1
        output.append(match.group("prefix") + normalized + _line_ending(line))

    return b"".join(output), change_count


def flatten_imports(
    content: bytes, public_names: frozenset[str] | None = None
) -> tuple[bytes, int]:
    """Flatten public imports, deduplicate star imports, and expand them.

    `public_names` is qmcpy's public API surface (see `_load_qmcpy_public_names`).
    When it's None, star imports are still deduplicated but left unexpanded.
    """

    change_count = 0

    def replace(match: re.Match[bytes]) -> bytes:
        nonlocal change_count
        module_segments = match.group("module_path").lstrip(b".").split(b".")
        if any(segment.startswith(b"_") for segment in module_segments):
            return match.group(0)
        if PRIVATE_NAME_RE.search(match.group("imported")):
            return match.group(0)

        change_count += 1
        return (
            b"from"
            + match.group("after_from")
            + b"qmcpy"
            + match.group("before_import")
            + b"import"
            + match.group("imported")
        )

    updated = QMCPY_IMPORT_RE.sub(replace, content)
    updated, duplicate_count = _deduplicate_adjacent_star_imports(updated)
    change_count += duplicate_count

    if public_names and STAR_IMPORT_LITERAL in updated:
        expand = (
            _expand_notebook_star_imports
            if _notebook_python_source(updated) is not None
            else _expand_text_star_imports
        )
        updated, expand_count = expand(updated, public_names)
        change_count += expand_count

    updated, comma_count = _normalize_comma_spacing(updated)
    change_count += comma_count

    return updated, change_count


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def iter_target_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield supported files under paths, pruning generated and cache directories."""

    seen: set[Path] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

        if path.is_file():
            if not _is_supported(path):
                raise ValueError(f"unsupported file type: {path}")
            candidates = [path]
        else:
            candidates = []
            for root, directory_names, file_names in os.walk(path):
                directory_names[:] = sorted(
                    name
                    for name in directory_names
                    if name not in SKIPPED_DIRECTORIES
                    and not name.endswith((".egg-info", ".dist-info"))
                )
                root_path = Path(root)
                candidates.extend(
                    root_path / name
                    for name in sorted(file_names)
                    if _is_supported(Path(name))
                )

        for candidate in candidates:
            if candidate.is_symlink():
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def _display_path(path: Path, base: Path) -> Path:
    try:
        return path.resolve().relative_to(base)
    except ValueError:
        return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report files that need changes without rewriting them",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="files or directories to process (default: repository root)",
    )
    args = parser.parse_args(argv)

    repository_root = Path(__file__).resolve().parent.parent
    paths = args.paths or [repository_root]

    try:
        targets = sorted(iter_target_files(paths), key=str)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    public_names = _load_qmcpy_public_names(repository_root)
    if public_names is None:
        print(
            "warning: qmcpy is not importable; star imports will be "
            "deduplicated but not expanded",
            file=sys.stderr,
        )

    changed_files = 0
    changed_imports = 0
    for path in targets:
        original = path.read_bytes()
        updated, count = flatten_imports(original, public_names)
        if not count:
            continue

        changed_files += 1
        changed_imports += count
        if not args.check:
            path.write_bytes(updated)
        action = "Would update" if args.check else "Updated"
        import_label = "import" if count == 1 else "imports"
        print(
            f"{action}: {_display_path(path, repository_root)} "
            f"({count} {import_label})"
        )

    if changed_files:
        action = "need updates" if args.check else "updated"
        import_label = "import" if changed_imports == 1 else "imports"
        file_label = "file" if changed_files == 1 else "files"
        print(
            f"{changed_imports} {import_label} in "
            f"{changed_files} {file_label} {action}."
        )
    else:
        print("All eligible QMCPy imports already use the top-level package.")

    return int(args.check and changed_files > 0)


if __name__ == "__main__":
    raise SystemExit(main())
