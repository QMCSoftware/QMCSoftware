"""
Unified VS Code + Colab bootstrap header (auto-detects notebook path).

Usage in a notebook:

    # Remote run in Colab (from your repo):
    %run https://raw.githubusercontent.com/QMCSoftware/QMCSoftware/bootstrap_colab/utils/notebook_header.py
    # (after merge, change branch above to "develop")

    # Local run in VS Code (from repo root):
    %run utils/notebook_header.py
"""

from __future__ import annotations
import os
import sys
import pathlib
import subprocess
from typing import Optional

# --- Self-switching bootstrap ---
# If running inside Colab but the file is being loaded locally, reload it from GitHub
import sys
if "google.colab" in sys.modules and not __file__.startswith("/content/"):
    import urllib.request, pathlib
    branch = "bootstrap_colab"  # change to develop after merge
    url = f"https://raw.githubusercontent.com/QMCSoftware/QMCSoftware/{branch}/utils/notebook_header.py"
    dest = pathlib.Path("/content/notebook_header.py")
    print(f"[notebook_header] Re-fetching from GitHub: {url}")
    urllib.request.urlretrieve(url, dest)
    get_ipython().run_line_magic("run", str(dest))
    raise SystemExit  # stop running the local file

# ---- One place to switch branches later ----
BOOT_BRANCH = os.environ.get("BOOT_BRANCH", "bootstrap_colab")  # change to "develop" after merge

IN_COLAB = "google.colab" in sys.modules
REPO = "QMCSoftware"
ORG = "QMCSoftware"


def _git_repo_root() -> Optional[pathlib.Path]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        p = pathlib.Path(out)
        return p if p.exists() else None
    except Exception:
        return None


def _guess_nb_path(repo_root: pathlib.Path) -> Optional[str]:
    """
    Best-effort notebook path detection relative to repo root.

    Tries (in order):
      1) ipynbname (no install, only if already available)
      2) If exactly one .ipynb exists in CWD, use it
      3) If the Python file name ends with .ipynb (rare in VS Code), use it
    """
    # 1) ipynbname (only if already installed)
    try:
        import ipynbname  # type: ignore
        nb_path = pathlib.Path(ipynbname.path())  # absolute
        try:
            return str(nb_path.relative_to(repo_root))
        except Exception:
            return str(nb_path)
    except Exception:
        pass

    # 2) exactly one .ipynb in current directory
    cwd = pathlib.Path.cwd()
    ipynbs = list(cwd.glob("*.ipynb"))
    if len(ipynbs) == 1:
        try:
            return str(ipynbs[0].resolve().relative_to(repo_root))
        except Exception:
            return str(ipynbs[0].name)

    # 3) Fallback: sometimes VS Code exposes __file__-like info (rare)
    try:
        import __main__  # type: ignore
        fn = getattr(__main__, "__file__", None)
        if isinstance(fn, str) and fn.endswith(".ipynb"):
            p = pathlib.Path(fn).resolve()
            try:
                return str(p.relative_to(repo_root))
            except Exception:
                return str(p)
    except Exception:
        pass

    return None


def _run_bootstrap(branch: str) -> None:
    """Download bootstrap_colab.py from given branch and run it (Colab only)."""
    import urllib.request
    url = f"https://raw.githubusercontent.com/{ORG}/{REPO}/{branch}/bootstrap_colab.py"
    dest = pathlib.Path("/content/bootstrap_colab.py")
    print(f"[notebook_header] Fetching: {url}")
    urllib.request.urlretrieve(url, dest)  # raises on 404
    print(f"[notebook_header] Saved to: {dest}")
    get_ipython().run_line_magic("run", str(dest))


def _show_colab_button(nb_path: Optional[str], branch: str) -> None:
    try:
        from utils.colab_button import show
    except Exception as e:
        print("[notebook_header] Colab button unavailable:", e)
        return

    if nb_path:
        show(ORG, REPO, nb_path, branch=branch, new_tab=False)
    else:
        print("[notebook_header] Could not determine NB_PATH automatically.")
        print("  Tip: set an explicit path at the very top of the notebook, then re-run:")
        print('    NB_PATH = "relative/path/to/this_notebook.ipynb"')
        print("  (from the repo root)")


def main() -> None:
    if IN_COLAB:
        # Ensure the package install matches the same branch as this header
        os.environ["QMCSOFTWARE_REF"] = BOOT_BRANCH
        _run_bootstrap(BOOT_BRANCH)
        return

    # Local (VS Code / Jupyter)
    repo_root = _git_repo_root()
    if not repo_root:
        print("[notebook_header] Not inside a git repo; skipping Colab button.")
        return

    nb_path = _guess_nb_path(repo_root)
    _show_colab_button(nb_path, BOOT_BRANCH)


if __name__ == "__main__":
    main()