"""
notebook_header.py — portable header for VS Code, JupyterLab, and Colab.

- Local: ensures <repo_root> and <repo_root>/utils are on sys.path.
- Colab: clones repo (+ submodules), installs minimal deps, ensures LaTeX toolchain,
         and makes qmcpy importable from the qmcsoftware submodule (or installs from GitHub).
- Always shows an "Open in Colab" badge (inline).
- Auto-imports (np/pd/plt/sp/sy/qp) + plotting prefs via utils/auto_imports.py.
- Colab-only execution time/timestamp hook (no duplicates in VS Code/JupyterLab).
- Quiet by default; set AUTO_IMPORTS_VERBOSE=1 for light diagnostics.

Env to set in first cell BEFORE importing this module:
  BOOT_ORG, BOOT_REPO, NB_PATH, BOOT_BRANCH, NOTEBOOK_HEADER_AUTORUN, AUTO_PLOT_PREFS
"""

from __future__ import annotations
import os
import sys
import re
import subprocess
import pathlib
import shutil
import time
import datetime
from typing import Optional, Tuple

# ---------------- State / Defaults ----------------
_STATE = {"ran": False}
_NB_PATH = "unknown.ipynb"

DEFAULT_BOOT_BRANCH = (
    os.environ.get("BOOT_BRANCH")
    or os.environ.get("QMCSOFTWARE_REF")
    or "bootstrap_colab"
)

# ---------------- Basic env helpers ----------------
def in_ipython() -> bool:
    try:
        get_ipython  # type: ignore
        return True
    except NameError:
        return False

def in_colab() -> bool:
    return ("COLAB_RELEASE_TAG" in os.environ) or ("COLAB_GPU" in os.environ)

# ---------------- Repo / path helpers ----------------
def get_repo_root() -> Optional[pathlib.Path]:
    try:
        p = pathlib.Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip())
        return p if p.exists() else None
    except Exception:
        return None

def get_org_repo() -> Tuple[str, str]:
    # Prefer explicit env overrides
    org = os.environ.get("BOOT_ORG")
    repo = os.environ.get("BOOT_REPO")
    if org and repo:
        return str(org), str(repo)
    # Fallback: try git remote
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True
        ).strip()
        m = re.search(r"[:/](?P<org>[^/]+)/(?P<repo>[^/\.]+)(?:\.git)?$", url)
        if m:
            return m.group("org"), m.group("repo")
    except Exception:
        pass
    return "QMCSoftware", "QMCSoftware"

def _resolve_branch(org: str, repo: str) -> str:
    env_branch = os.environ.get("BOOT_BRANCH")
    if env_branch:
        return env_branch
    if repo == "MATH565Fall2025":
        return "main"
    return DEFAULT_BOOT_BRANCH

def get_nb_override() -> Optional[str]:
    # user_ns NB_PATH takes precedence
    try:
        if in_ipython():
            val = get_ipython().user_ns.get("NB_PATH")  # type: ignore
            if isinstance(val, str) and val.strip():
                return val
    except Exception:
        pass
    val = os.environ.get("NB_PATH")
    return val if val else None

def guess_nb_path(repo_root: pathlib.Path) -> str:
    try:
        import ipynbname  # type: ignore
        p = pathlib.Path(ipynbname.path()).resolve()
        return str(p.relative_to(repo_root))
    except Exception:
        return "unknown.ipynb"

def _ensure_local_paths(repo_root: pathlib.Path) -> None:
    for p in (repo_root, repo_root / "utils"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

# ---------------- Colab helpers ----------------
def ensure_pip_packages(pkgs: list, quiet: bool = True) -> None:
    import importlib
    to_install = []
    for pkg in pkgs:
        mod = pkg.split("==")[0].split(">=")[0]
        try:
            importlib.import_module(mod)
        except Exception:
            to_install.append(pkg)
    if to_install:
        if not quiet:
            print("[notebook_header] pip install:", " ".join(to_install))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])

def _have_tex_toolchain() -> bool:
    have_latex = shutil.which("latex") is not None
    have_dvipng = (shutil.which("dvipng") is not None) or (shutil.which("dvisvgm") is not None)
    have_gs = (shutil.which("gs") is not None) or (shutil.which("ghostscript") is not None)
    return have_latex and have_dvipng and have_gs

def ensure_latex_toolchain(quiet: bool = True) -> bool:
    """Colab: install LaTeX if missing (prints 'this may take several minutes'). Local: no install."""
    if _have_tex_toolchain():
        return True
    if in_colab():
        try:
            print("[notebook_header] LaTeX not found — installing in Colab (this may take several minutes)...")
            subprocess.check_call(["bash","-lc","apt-get -y update"])
            subprocess.check_call([
                "bash","-lc",
                "apt-get -y install texlive-latex-extra texlive-fonts-recommended dvipng dvisvgm cm-super ghostscript"
            ])
            return _have_tex_toolchain()
        except Exception as e:
            print("[notebook_header] WARNING: LaTeX install failed; Matplotlib usetex may not work:", e)
            return False
    # Local/JupyterLab/VS Code: user manages TeX; stay quiet by default
    if not quiet:
        print("[notebook_header] LaTeX toolchain not found locally. Please install TeX + dvipng/ghostscript.")
    return False

# ---- QMCSoftware submodule discovery / importability ----
def _find_qmc_submodule(repo_dir: pathlib.Path) -> Optional[pathlib.Path]:
    # Try common casings first
    for name in ("qmcsoftware", "QMCSoftware"):
        p = repo_dir / name
        if p.exists():
            return p
    # Fallback: scan for a folder containing 'qmcpy' (flat or src layout)
    try:
        for p in repo_dir.iterdir():
            if p.is_dir() and ((p / "qmcpy").exists() or (p / "src" / "qmcpy").exists()):
                return p
    except Exception:
        pass
    return None

def _is_editable_installable(path: pathlib.Path) -> bool:
    return any((path / f).exists() for f in ("pyproject.toml", "setup.cfg", "setup.py"))

def _add_qmcpy_to_syspath(qmc_path: pathlib.Path) -> None:
    # src-layout or flat
    cand = qmc_path / "src" if (qmc_path / "src" / "qmcpy").exists() else qmc_path
    sp = str(cand)
    if sp not in sys.path:
        sys.path.insert(0, sp)

def colab_bootstrap(org: str, repo: str, branch: str, quiet: bool = True) -> pathlib.Path:
    repo_dir = pathlib.Path("/content") / repo
    if not repo_dir.exists():
        if not quiet:
            print(f"[notebook_header] Cloning {org}/{repo}@{branch} (with submodules) ...")
        subprocess.check_call([
            "git","clone","--depth","1","--recurse-submodules",
            "-b", branch, f"https://github.com/{org}/{repo}.git", str(repo_dir)
        ])
    # Ensure submodules are present/up-to-date
    subprocess.check_call(["git","-C",str(repo_dir),"submodule","sync","--recursive"])
    subprocess.check_call(["git","-C",str(repo_dir),"submodule","update","--init","--recursive","--depth","1"])

    # Minimal Python deps needed by header utilities
    ensure_pip_packages(["numpy","scipy","matplotlib","pandas","ipynbname"], quiet=quiet)

    # Make qmcpy importable from submodule
    qmc_path = _find_qmc_submodule(repo_dir)
    if qmc_path:
        if _is_editable_installable(qmc_path):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(qmc_path)])
            except Exception as e:
                if not quiet:
                    print("[notebook_header] WARNING: pip -e for QMCSoftware failed; falling back to sys.path:", e)
                _add_qmcpy_to_syspath(qmc_path)
        else:
            _add_qmcpy_to_syspath(qmc_path)
    else:
        if not quiet:
            print("[notebook_header] WARNING: QMCSoftware submodule not found; will try GitHub fallback later.")

    # Add repo + utils to sys.path
    for p in (repo_dir, repo_dir / "utils"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    try:
        os.chdir(repo_dir)
    except Exception:
        pass
    return repo_dir

def ensure_qmcpy_from_github():
    """If qmcpy is not importable, install from GitHub (optional branch via QMCPY_BRANCH)."""
    try:
        import qmcpy  # noqa: F401
        return  # already available
    except Exception:
        pass

    branch = os.environ.get("QMCPY_BRANCH", "").strip()
    url = "git+https://github.com/QMCSoftware/QMCSoftware.git"
    if branch:
        url += f"@{branch}"
    # '#egg=qmcpy' is optional; useful for some older pip resolutions
    url += "#egg=qmcpy"

    print("[notebook_header] Installing qmcpy from GitHub"
          f"{' (branch: ' + branch + ')' if branch else ''} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", url])

# ---------------- Colab badge ----------------
from urllib.parse import quote

def show_colab_button(org: str, repo: str, branch: str, nb_path: str) -> None:
    from IPython.display import HTML, display
    nb_quoted = quote(nb_path, safe="/")
    url = f"https://colab.research.google.com/github/{org}/{repo}/blob/{branch}/{nb_quoted}"
    html = (
        '<div style="font-size:120%;">'
        'If not running in the <code>conda qmcpy</code> environment, <br>'
        '<span style="margin-left:1.5em;">'
        f'push the button to <a target="_blank" href="{url}">'
        '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
        ' and then run from the top. <br>Otherwise continue to the next cell.'
        '</div>'
    )
    display(HTML(html))

# ---------------- Auto-imports hook ----------------
def try_auto_imports() -> None:
    try:
        try:
            from utils.auto_imports import inject_common  # preferred path
            src = "utils.auto_imports"
        except ImportError:
            import auto_imports
            inject_common = auto_imports.inject_common
            src = "auto_imports"
        ns = get_ipython().user_ns  # type: ignore
        verbose = os.environ.get("AUTO_IMPORTS_VERBOSE", "0").lower() in ("1","true","yes")
        plot_prefs = os.environ.get("AUTO_PLOT_PREFS", "0").lower() in ("1","true","yes")
        inject_common(ns, verbose=verbose, plot_prefs=plot_prefs)
        if verbose:
            print(f"[notebook_header] auto_imports loaded from {src}")
    except Exception as e:
        print("[notebook_header] auto_imports failed:", e)

# ---------------- Colab-only execution timer ----------------
def _register_execution_timer() -> None:
    """Prints '[Executed in X.XXs at YYYY-MM-DD HH:MM:SS timezone]' after each cell (Colab only)."""
    if _STATE.get("timer"):
        return
    # (rest as-is)
    _STATE["timer"] = True
    
    if not in_ipython():
        return
    ip = get_ipython()  # type: ignore
    if not ip:
        return
    _state = {"start": None}

    def pre_run_cell(_info):
        _state["start"] = time.perf_counter()

    def post_run_cell(_result):
        if _state["start"] is None:
            return
        elapsed = time.perf_counter() - _state["start"]
        ts = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[Executed in {elapsed:.2f}s at {ts}]")

    # Register hooks (idempotent enough for our usage)
    ip.events.register("pre_run_cell", pre_run_cell)
    ip.events.register("post_run_cell", post_run_cell)

# ---------------- Main ----------------
def main(force: bool = False, quiet: bool = True):
    """Run header once; on subsequent calls, re-render the badge and exit."""
    global _NB_PATH

    repo_root = get_repo_root()
    org, repo = get_org_repo()
    branch = _resolve_branch(org, repo)

    # Local path wiring
    if repo_root and not in_colab():
        _ensure_local_paths(repo_root)

    if _STATE["ran"] and not force:
        # Re-render badge so re-running the cell doesn't blank it
        try:
            show_colab_button(org, repo, branch, _NB_PATH)
        except Exception:
            pass
        return
    _STATE["ran"] = True

    # Determine notebook path
    nb_override = get_nb_override()
    if nb_override:
        _NB_PATH = nb_override
    elif repo_root:
        _NB_PATH = guess_nb_path(repo_root)
    else:
        _NB_PATH = "unknown.ipynb"

    # Colab setup
    if in_colab():
        colab_bootstrap(org, repo, branch, quiet=quiet)
        ensure_latex_toolchain(quiet=True)
        ensure_qmcpy_from_github()
        _register_execution_timer()  # Colab-only timing hook

    # Auto-imports + badge
    try_auto_imports()
    try:
        show_colab_button(org, repo, branch, _NB_PATH)
    except Exception:
        pass

def reload_header(quiet: bool = True):
    return main(force=True, quiet=quiet)

# ---------------- Autorun ----------------
if os.environ.get("NOTEBOOK_HEADER_AUTORUN", "1").lower() in ("1", "true", "yes"):
    if in_ipython():
        try:
            main(False)
        except Exception as e:
            print("[notebook_header] ERROR: autorun failed:", e)