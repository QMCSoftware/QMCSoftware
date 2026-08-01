#!/usr/bin/env python3
"""Flatten public absolute QMCPy imports to use the top-level package."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
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


def flatten_imports(content: bytes) -> tuple[bytes, int]:
    """Flatten public imports while preserving imports involving private names."""

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

    return QMCPY_IMPORT_RE.sub(replace, content), change_count


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
        targets = sorted(iter_target_files(paths), key=lambda path: str(path))
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    changed_files = 0
    changed_imports = 0
    for path in targets:
        original = path.read_bytes()
        updated, count = flatten_imports(original)
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
