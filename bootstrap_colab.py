"""
Bootstrap for running QMCSoftware notebooks on Google Colab.

Usage (top of a notebook):

  # Change BOOT_BRANCH below to "develop" after merge
  BOOT_BRANCH = "bootstrap_colab"

  # Option A: default (no LaTeX)
  %run https://raw.githubusercontent.com/QMCSoftware/QMCSoftware/{BOOT_BRANCH}/bootstrap_colab.py

  # Option B: enable LaTeX via env var
  %env USE_TEX=1
  %run https://raw.githubusercontent.com/QMCSoftware/QMCSoftware/{BOOT_BRANCH}/bootstrap_colab.py

  # Option C: enable LaTeX later at runtime
  import bootstrap_colab
  bootstrap_colab.enable_latex()

Behavior:
- Detects Colab and installs a minimal scientific Python stack.
- Conditionally installs a LaTeX toolchain for matplotlib's usetex.
- Installs QMCSoftware from Git (read-only), pointing at a branch or commit SHA.
- No changes when running locally.
"""
from __future__ import annotations
import os
import sys
import subprocess
from typing import Iterable

# =====================
# Configuration toggles
# =====================
# Branch used for examples and default install — change after merge
BOOT_BRANCH = "bootstrap_colab"

# Read from environment so notebooks can set: %env USE_TEX=1
USE_TEX: bool = os.environ.get("USE_TEX", "0") == "1"

# Which ref of QMCSoftware to install. Defaults to BOOT_BRANCH unless overridden by env.
QMCSOFTWARE_REF: str = os.environ.get("QMCSOFTWARE_REF", BOOT_BRANCH)

# When True, installs QMCSoftware from Git (non-editable, read-only for students).
INSTALL_QMCSOFTWARE: bool = os.environ.get("INSTALL_QMCSOFTWARE", "1") == "1"

# Minimal scientific stack to ensure is present on Colab
REQUIRED_PYPI: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
)

_LATEX_ENABLED = False


def _in_colab() -> bool:
    return "google.colab" in sys.modules


def _run(cmd: list[str] | str) -> int:
    """Run a shell command quietly but return exit code (prints on failure)."""
    try:
        if isinstance(cmd, str):
            code = subprocess.call(cmd, shell=True)
        else:
            code = subprocess.call(cmd)
        if code != 0:
            print(f"Command failed (exit {code}): {cmd}")
        return code
    except Exception as e:
        print(f"Error running {cmd}: {e}")
        return 1


def _pip_install(pkgs: Iterable[str]) -> None:
    pkgs = tuple(pkgs)
    if not pkgs:
        return
    _run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "pip>=24.0"])
    _run([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def _install_latex_toolchain() -> None:
    print("Installing LaTeX packages for matplotlib usetex… (this is ~500MB, 1–2 minutes)")
    _run("apt-get update -qq")
    _run("apt-get install -y -qq cm-super dvipng texlive-latex-extra texlive-latex-recommended")


def _ensure_scientific_stack() -> None:
    print("Ensuring scientific Python stack…")
    _pip_install(REQUIRED_PYPI)


def _install_qmcsoftware(ref: str) -> None:
    """Install QMCSoftware from Git into site-packages (non-editable)."""
    url = f"git+https://github.com/QMCSoftware/QMCSoftware@{ref}"
    print(f"Installing QMCSoftware from {url} …")
    _pip_install((url,))
    try:
        import importlib.metadata as md  # py3.8+
        ver = None
        try:
            ver = md.version("qmcpy")
        except md.PackageNotFoundError:
            ver = None
        if ver:
            print("qmcpy installed, version:", ver)
        else:
            import qmcpy  # type: ignore
            print("qmcpy installed, version:", getattr(qmcpy, "__version__", "unknown"))
    except Exception as e:
        print("qmcpy import check failed:", e)


def enable_latex() -> None:
    """Install LaTeX toolchain (if in Colab) and enable matplotlib usetex at runtime."""
    global _LATEX_ENABLED
    if _LATEX_ENABLED:
        print("LaTeX already enabled.")
        return
    if not _in_colab():
        print("Not in Colab; enable LaTeX in your local environment as needed.")
        return
    _install_latex_toolchain()
    try:
        import matplotlib as mpl
        mpl.rcParams["text.usetex"] = True
        _LATEX_ENABLED = True
        print("Enabled matplotlib usetex.")
    except Exception as e:
        print("Could not set matplotlib usetex:", e)


def main() -> None:
    if not _in_colab():
        print("Running locally — using your existing environment (no changes).")
        return

    print("Running in Colab — setting up environment…")
    _ensure_scientific_stack()

    if USE_TEX:
        enable_latex()
    else:
        print("Using matplotlib mathtext (no LaTeX install).")

    if INSTALL_QMCSOFTWARE:
        _install_qmcsoftware(QMCSOFTWARE_REF)
    else:
        print("Skipped installing QMCSoftware (INSTALL_QMCSOFTWARE=0).")


if __name__ == "__main__":
    main()