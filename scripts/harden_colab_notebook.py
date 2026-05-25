#!/usr/bin/env python3
"""
Harden a demo notebook for Colab and classify it in the Colab manifest.

This script is intentionally conservative:
- it updates one notebook at a time
- it inserts a standard badge/bootstrap before the first substantive code cell
- it only auto-classifies notebooks that are not already in enabled or disabled
- it can force-regenerate Colab bootstrap cells for notebooks already in enabled
- it validates the result with the existing strict Colab checks
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from check_colab_notebooks import (
    DEFAULT_MANIFEST,
    EXTRA_PIP_DEPENDENCIES,
    REPO_ROOT,
    badge_markup,
    discovered_notebooks,
    early_non_install_code_cells,
    imported_modules,
    is_any_install_cell,
    local_module_matches,
    load_json,
    validate_disabled_notebook,
    validate_enabled_notebook,
    validate_manifest,
    manifest_sets,
    validate_strict_enabled_notebook,
)


EXTRA_IMPORT_DEPENDENCIES = {
    **EXTRA_PIP_DEPENDENCIES,
    "botorch": ("botorch",),
    "gpytorch": ("gpytorch",),
    "ipywidgets": ("ipywidgets",),
    "seaborn": ("seaborn",),
    "sklearn": ("scikit-learn", "sklearn"),
    "sympy": ("sympy",),
    "torch": ("torch",),
    "umbridge": ("umbridge",),
    "yfinance": ("yfinance",),
}
LATEX_MARKERS = (
    "text.usetex",
    "\\usepackage",
    "dvipng",
    "latexmk",
    "computer modern",
)


def dump_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


def as_source_text(source: str | list[str]) -> str:
    if isinstance(source, str):
        return source
    return "".join(source)


def full_source(cells: list[dict], cell_types: set[str] | None = None) -> str:
    if cell_types is None:
        cell_types = {"code"}
    return "\n".join(
        as_source_text(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") in cell_types
    )


def discovered_imports(cells: list[dict]) -> set[str]:
    modules: set[str] = set()
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        modules.update(
            imported_modules(
                as_source_text(cell.get("source", [])),
                location=f"cell {idx + 1}",
            )
        )
    return modules


def local_repo_import_matches(
    notebook_path: Path, cells: list[dict]
) -> dict[str, list[Path]]:
    notebook_dir = notebook_path.parent.resolve()
    matches_by_module: dict[str, list[Path]] = {}
    for module in sorted(discovered_imports(cells)):
        if module == "qmcpy":
            continue
        matches = local_module_matches(notebook_dir, module)
        if matches:
            matches_by_module[module] = matches
    return matches_by_module


def needs_latex_setup(cells: list[dict]) -> bool:
    source = full_source(cells, {"code", "markdown"})
    return any(marker in source for marker in LATEX_MARKERS)


def extra_pip_packages(cells: list[dict]) -> list[str]:
    early_cells = [cell for _, cell in early_non_install_code_cells(cells)]
    modules = discovered_imports(early_cells)
    packages: list[str] = []
    for module, names in sorted(EXTRA_IMPORT_DEPENDENCIES.items()):
        if module in modules:
            package = names[0]
            if package not in packages:
                packages.append(package)
    return packages


def extra_repo_paths(notebook_path: Path, cells: list[dict]) -> list[str]:
    notebook_dir = notebook_path.parent.resolve()
    repo_matches = local_repo_import_matches(notebook_path, cells)
    rel_paths: list[str] = []

    for module in sorted(repo_matches):
        for match in repo_matches[module]:
            if match.name == "__init__.py" and match.parent.name == module:
                parent = match.parent.parent.resolve()
            else:
                parent = match.parent.resolve()
            if parent == notebook_dir:
                continue
            if parent == REPO_ROOT:
                continue
            rel_parent = parent.relative_to(REPO_ROOT).as_posix()
            if rel_parent not in rel_paths:
                rel_paths.append(rel_parent)

    return rel_paths


def bootstrap_cell_source(notebook_path: Path, manifest: dict, cells: list[dict]) -> list[str]:
    notebook_dir_rel = notebook_path.parent.relative_to(REPO_ROOT).as_posix()
    packages = extra_pip_packages(cells)
    rel_paths = extra_repo_paths(notebook_path, cells)
    latex_setup = needs_latex_setup(cells)
    needs_repo_clone = bool(local_repo_import_matches(notebook_path, cells))

    lines = [
        "# @title Execute this cell to install dependencies\n",
        "try:\n",
        "  import google.colab\n",
    ]

    if needs_repo_clone:
        lines.extend(
            [
                "  import sys\n",
                "  import os\n",
                '  repo_root = "/content/QMCSoftware"\n',
                f'  notebook_dir = f"{{repo_root}}/{notebook_dir_rel}"\n',
                "  if not os.path.isdir(repo_root):\n",
                f"    !git clone -q --depth 1 https://github.com/{manifest['repo']} {{repo_root}}\n",
            ]
        )

    lines.append("  !pip install -q qmcpy\n")

    if packages:
        lines.append(f"  !pip install -q {' '.join(packages)}\n")

    if latex_setup:
        lines.append(
            "  !apt-get update && apt-get install -y -qq --no-install-recommends texlive-latex-base texlive-fonts-recommended texlive-latex-extra cm-super dvipng\n"
        )

    if needs_repo_clone:
        lines.extend(
            [
                "  os.chdir(notebook_dir)\n",
                "  if notebook_dir not in sys.path:\n",
                "    sys.path.insert(0, notebook_dir)\n",
            ]
        )
        for rel_path in rel_paths:
            lines.extend(
                [
                    f'  extra_path = f"{{repo_root}}/{rel_path}"\n',
                    "  if extra_path not in sys.path:\n",
                    "    sys.path.insert(0, extra_path)\n",
                ]
            )

    lines.extend(["except:\n", "  pass"])
    return lines


def first_code_index(cells: list[dict]) -> int:
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            return idx
    return len(cells)


def badge_exists(cells: list[dict], manifest: dict, notebook_path: Path) -> bool:
    expected = badge_markup(
        manifest["repo"],
        manifest["git_ref"],
        notebook_path.relative_to(REPO_ROOT).as_posix(),
    )
    return any(
        cell.get("cell_type") == "markdown"
        and expected in as_source_text(cell.get("source", []))
        for cell in cells
    )


def remove_any_badge_cells(cells: list[dict]) -> list[dict]:
    return [
        cell
        for cell in cells
        if not (
            cell.get("cell_type") == "markdown"
            and "colab.research.google.com" in as_source_text(cell.get("source", []))
        )
    ]


def remove_existing_bootstrap_cells(cells: list[dict]) -> list[dict]:
    return [cell for cell in cells if not is_any_install_cell(cell)]


def pending_unclassified(
    manifest: dict, current_notebook: str | None = None
) -> set[str]:
    enabled, disabled = manifest_sets(manifest)
    missing = discovered_notebooks() - enabled - set(disabled)
    if current_notebook is not None:
        missing.discard(current_notebook)
    return missing


def harden_notebook(notebook_path: Path, manifest_path: Path) -> None:
    manifest = load_json(manifest_path)
    notebook_payload = load_json(notebook_path)

    original_manifest = copy.deepcopy(manifest)
    original_notebook = copy.deepcopy(notebook_payload)

    # Cells that survive unchanged: remove only the old badge/bootstrap cells.
    # These dict objects are never modified; they are passed through as-is.
    kept_cells = remove_any_badge_cells(
        remove_existing_bootstrap_cells(list(notebook_payload.get("cells", [])))
    )

    insert_at = 1 if kept_cells and kept_cells[0].get("cell_type") == "markdown" else 0
    insert_at = min(insert_at, first_code_index(kept_cells))

    badge_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            badge_markup(
                manifest["repo"],
                manifest["git_ref"],
                notebook_path.relative_to(REPO_ROOT).as_posix(),
            )
        ],
    }
    bootstrap_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        # Pass kept_cells (not the final list) so the source scanner only sees
        # the original notebook code cells, not the badge cell we just built.
        "source": bootstrap_cell_source(notebook_path, manifest, kept_cells),
    }

    # Build the final cell list purely by concatenation — kept_cells are untouched.
    cells = kept_cells[:insert_at] + [badge_cell, bootstrap_cell] + kept_cells[insert_at:]

    notebook_payload["cells"] = cells

    notebook_rel = notebook_path.relative_to(REPO_ROOT).as_posix()
    enabled = list(manifest.get("enabled", []))
    disabled = dict(manifest.get("disabled", {}))
    if notebook_rel in disabled:
        raise ValueError(
            f"{notebook_rel} is already classified as disabled; harden_colab_notebook does not reclassify disabled notebooks."
        )
    if notebook_rel not in enabled:
        enabled.append(notebook_rel)
    enabled = sorted(enabled)
    manifest["enabled"] = enabled
    manifest["disabled"] = disabled

    try:
        dump_json(notebook_path, notebook_payload)
        dump_json(manifest_path, manifest)

        reloaded_manifest = load_json(manifest_path)
        errors = validate_manifest(
            reloaded_manifest,
            allowed_missing=pending_unclassified(reloaded_manifest, notebook_rel),
        )
        errors.extend(
            validate_enabled_notebook(
                notebook_path,
                reloaded_manifest["repo"],
                reloaded_manifest["git_ref"],
            )
        )
        errors.extend(validate_strict_enabled_notebook(notebook_path))
        if errors:
            raise RuntimeError("\n".join(errors))
    except Exception:
        dump_json(notebook_path, original_notebook)
        dump_json(manifest_path, original_manifest)
        raise


def unclassified_notebooks(manifest_path: Path) -> list[Path]:
    manifest = load_json(manifest_path)
    enabled, disabled = manifest_sets(manifest)
    unclassified = sorted(discovered_notebooks() - enabled - set(disabled))
    notebook_paths: list[Path] = []
    for notebook_rel in unclassified:
        notebook_paths.append((REPO_ROOT / notebook_rel).resolve())
    return notebook_paths


def enabled_notebooks(manifest_path: Path) -> list[Path]:
    manifest = load_json(manifest_path)
    enabled, _ = manifest_sets(manifest)
    return sorted((REPO_ROOT / notebook_rel).resolve() for notebook_rel in enabled)


def disable_notebook(notebook_path: Path, manifest_path: Path, reason: str) -> None:
    manifest = load_json(manifest_path)
    original_manifest = copy.deepcopy(manifest)
    notebook_payload = load_json(notebook_path)
    original_notebook = copy.deepcopy(notebook_payload)
    notebook_rel = notebook_path.relative_to(REPO_ROOT).as_posix()
    enabled = sorted(
        path for path in manifest.get("enabled", []) if path != notebook_rel
    )
    manifest["enabled"] = enabled
    disabled = dict(manifest.get("disabled", {}))
    disabled[notebook_rel] = reason
    manifest["disabled"] = disabled
    try:
        notebook_payload["cells"] = remove_existing_bootstrap_cells(
            remove_any_badge_cells(list(notebook_payload.get("cells", [])))
        )
        dump_json(notebook_path, notebook_payload)
        dump_json(manifest_path, manifest)
        reloaded_manifest = load_json(manifest_path)
        errors = validate_manifest(
            reloaded_manifest,
            allowed_missing=pending_unclassified(reloaded_manifest, notebook_rel),
        )
        errors.extend(validate_disabled_notebook(notebook_path))
        if errors:
            raise RuntimeError("\n".join(errors))
    except Exception:
        dump_json(notebook_path, original_notebook)
        dump_json(manifest_path, original_manifest)
        raise


def harden_unclassified_notebooks(
    manifest_path: Path,
) -> tuple[list[str], list[tuple[str, str]]]:
    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for notebook_path in unclassified_notebooks(manifest_path):
        notebook_rel = notebook_path.relative_to(REPO_ROOT).as_posix()
        try:
            if not notebook_path.exists():
                raise FileNotFoundError(f"Notebook not found: {notebook_rel}")
            if notebook_path.suffix != ".ipynb":
                raise ValueError(f"Expected a notebook path ending in .ipynb: {notebook_rel}")
            harden_notebook(notebook_path, manifest_path)
            successes.append(notebook_rel)
        except Exception as exc:
            summary = str(exc).splitlines()[0] if str(exc) else "unknown error"
            disable_notebook(
                notebook_path,
                manifest_path,
                f"Automatic Colab hardening failed: {summary}",
            )
            failures.append((notebook_rel, summary))

    return successes, failures


def harden_enabled_notebooks(
    manifest_path: Path,
) -> tuple[list[str], list[tuple[str, str]]]:
    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for notebook_path in enabled_notebooks(manifest_path):
        notebook_rel = notebook_path.relative_to(REPO_ROOT).as_posix()
        try:
            harden_notebook(notebook_path, manifest_path)
            successes.append(notebook_rel)
        except Exception as exc:
            summary = str(exc).splitlines()[0] if str(exc) else "unknown error"
            failures.append((notebook_rel, summary))

    return successes, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harden a demo notebook for Colab and enable it in the Colab manifest."
    )
    parser.add_argument(
        "--notebook",
        help="Repo-relative path to the notebook to harden, e.g. demos/plot_proj_function.ipynb",
    )
    parser.add_argument(
        "--all-unclassified",
        action="store_true",
        help="Attempt to harden every notebook under demos/ that is not yet listed in enabled or disabled.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the Colab bootstrap for enabled notebook(s). Without --notebook, processes every enabled notebook.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"Path to the Colab manifest (default: {DEFAULT_MANIFEST})",
    )
    args = parser.parse_args()
    if args.notebook and args.all_unclassified:
        parser.error("specify either --notebook or --all-unclassified, not both")
    if args.all_unclassified and args.force:
        parser.error("specify either --all-unclassified or --force, not both")
    if not args.notebook and not args.all_unclassified and not args.force:
        parser.error("specify --notebook, --all-unclassified, or --force")
    return args


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    if args.notebook:
        notebook_path = (REPO_ROOT / args.notebook).resolve()
        if not notebook_path.exists():
            raise SystemExit(f"Notebook not found: {args.notebook}")
        if notebook_path.suffix != ".ipynb":
            raise SystemExit(f"Expected a notebook path ending in .ipynb: {args.notebook}")
        manifest = load_json(manifest_path)
        enabled, disabled = manifest_sets(manifest)
        notebook_rel = notebook_path.relative_to(REPO_ROOT).as_posix()
        if notebook_rel in disabled:
            raise SystemExit(
                f"{notebook_rel} is already classified as disabled; update the manifest manually before reclassifying it."
            )
        if notebook_rel in enabled:
            harden_notebook(notebook_path, manifest_path)
            print(f"Hardened {notebook_rel} for Colab.")
            return 0
        try:
            harden_notebook(notebook_path, manifest_path)
            print(f"Hardened {notebook_rel} for Colab and added it to enabled.")
            return 0
        except Exception as exc:
            summary = str(exc).splitlines()[0] if str(exc) else "unknown error"
            disable_notebook(
                notebook_path,
                manifest_path,
                f"Automatic Colab hardening failed: {summary}",
            )
            print(
                f"Could not harden {notebook_rel}; restored the notebook and added it to disabled."
            )
            print(f"Reason: {summary}")
            return 1

    if args.force:
        successes, failures = harden_enabled_notebooks(manifest_path)
    else:
        successes, failures = harden_unclassified_notebooks(manifest_path)
    for notebook_rel in successes:
        print(f"Hardened {notebook_rel} for Colab.")
    if failures:
        print("")
        print("Not yet hardened:")
        for notebook_rel, error in failures:
            summary = error.splitlines()[0] if error else "unknown error"
            print(f"- {notebook_rel}: {summary}")
    print("")
    print(
        f"Hardened {len(successes)} notebook(s); {len(failures)} notebook(s) still need manual follow-up."
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
