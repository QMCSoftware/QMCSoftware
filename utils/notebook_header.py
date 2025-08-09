"""
notebook_header.py — portable header for VS Code, JupyterLab, and Colab.

Features:
- Location-independent: detects repo root, org/repo, and notebook path.
- Honors NB_PATH override set in the notebook or environment.
- Colab bootstrap: clones current repo (+ submodules) and installs deps.
- Colab badge: shows "Open in Colab" when not in 'qmcpy' env (or always in Colab).
- Post-cell hook: if NB path is unknown on first import, resolve it right after the first cell.
- Auto-run on import; idempotent; re-run via reload_header().
- quiet=True by default to suppress print output.
"""

from __future__ import annotations
import os
import sys
import re
import subprocess
import pathlib
import datetime
from typing import Optional, List

_STATE = {"ran": False}
_NB_PATH = "unknown.ipynb"  # updated at runtime

DEFAULT_BOOT_BRANCH = (
    os.environ.get("BOOT_BRANCH")
    or os.environ.get("QMCSOFTWARE_REF")
    or "bootstrap_colab"
)

# ---------------------------
# Environment helpers
# ---------------------------
def in_ipython() -> bool:
    try:
        get_ipython  # type: ignore[name-defined]
        return True
    except NameError:
        return False

def in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ

def conda_env_name() -> Optional[str]:
    name = os.environ.get("CONDA_DEFAULT_ENV")
    if name:
        return name
    prefix = sys.prefix
    m = re.search(r"(?:^|[/\\\\])envs[/\\\\]([^/\\\\]+)$", prefix)
    return m.group(1) if m else None

def in_qmcpy_env() -> bool:
    env = (conda_env_name() or "").lower()
    if env == "qmcpy":
        return True
    flag = os.environ.get("QMCPY_ENV", "").lower()
    if flag in ("1", "true", "yes", "y"):
        return True
    return False

# ---------------------------
# Git / repo helpers
# ---------------------------
def get_repo_root() -> Optional[pathlib.Path]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        p = pathlib.Path(out)
        return p if p.exists() else None
    except Exception:
        return None

def get_org_repo() -> tuple[str, str]:
    org = globals().get("ORG") or os.environ.get("ORG")
    repo = globals().get("REPO") or os.environ.get("REPO")
    if org and repo:
        return str(org), str(repo)
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True
        ).strip()
        m = re.search(r"[:/](?P<org>[^/]+)/(?P<repo>[^/\\.]+)(?:\\.git)?$", url)
        if m:
            return m.group("org"), m.group("repo")
    except Exception:
        pass
    return "QMCSoftware", "QMCSoftware"

# ---------------------------
# NB_PATH override + detection
# ---------------------------
def get_nb_override() -> Optional[str]:
    try:
        if in_ipython():
            ns = get_ipython().user_ns  # type: ignore[attr-defined]
            val = ns.get("NB_PATH")
            if isinstance(val, str) and val.strip():
                return val
    except Exception:
        pass
    val = os.environ.get("NB_PATH")
    return val if val else None

def guess_nb_path(repo_root: pathlib.Path) -> str:
    try:
        import ipynbname  # type: ignore
        full = pathlib.Path(ipynbname.path())
        return str(full.resolve().relative_to(repo_root))
    except Exception:
        return "unknown.ipynb"

# ---------------------------
# Colab bootstrap
# ---------------------------
def ensure_pip_packages(pkgs: List[str]) -> None:
    import importlib
    to_install = []
    for p in pkgs:
        mod = p.split("==")[0].split(">=")[0]
        try:
            importlib.import_module(mod if mod != "matplotlib" else "matplotlib")
        except Exception:
            to_install.append(p)
    if to_install:
        print("[notebook_header] pip install:", " ".join(to_install))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])

def colab_bootstrap(org: str, repo: str, branch: str) -> pathlib.Path:
    repo_dir = pathlib.Path("/content") / repo
    if not repo_dir.exists():
        print(f"[notebook_header] Cloning {org}/{repo}@{branch} (with submodules) ...")
        subprocess.check_call([
            "git", "clone", "--depth", "1", "--recurse-submodules",
            "-b", branch, f"https://github.com/{org}/{repo}.git", str(repo_dir)
        ])
    else:
        print(f"[notebook_header] Using existing clone at {repo_dir}")

    req = repo_dir / "requirements-colab.txt"
    if req.exists():
        print("[notebook_header] Installing requirements-colab.txt")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])
    else:
        ensure_pip_packages([
            "numpy", "scipy", "matplotlib", "pandas", "ipynbname"
        ])

    qmc_path = repo_dir / "QMCSoftware"
    if qmc_path.exists():
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(qmc_path)])
        except Exception as e:
            print("[notebook_header] Editable install of QMCSoftware failed:", e)

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(repo_dir)])
    except Exception as e:
        print("[notebook_header] Editable install of repo failed:", e)

    utils_dir = repo_dir / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))

    try:
        os.chdir(repo_dir)
    except Exception:
        pass

    return repo_dir

# ---------------------------
# Colab badge helper
# ---------------------------
def show_colab_button(org: str, repo: str, branch: str, nb_path: str) -> None:
    from IPython.display import HTML, display
    nb_quoted = nb_path.replace(" ", "%20")
    url = f"https://colab.research.google.com/github/{org}/{repo}/blob/{branch}/{nb_quoted}"
    html = (
        'If you are not running this notebook in the '
        '<code>conda qmcpy</code> environment, '
        f'<a target="_blank" href="{url}">'
        '<img src="https://colab.research.google.com/assets/colab-badge.svg" '
        'alt="Open In Colab"/></a>'
        ', otherwise continue to the next cell for setup.'
    )
    display(HTML(html))

# ---------------------------
# Auto imports (optional)
# ---------------------------
def try_auto_imports() -> None:
    try:
        from utils.auto_imports import inject_common  # type: ignore
        inject_common(get_ipython().user_ns)  # type: ignore[attr-defined]
    except Exception:
        pass

# ---------------------------
# Post-run hook
# ---------------------------
def _post_run_attempt_path(repo_root: pathlib.Path, org: str, repo: str, branch: str, quiet: bool):
    ip = get_ipython()  # type: ignore[attr-defined]

    def _callback(*args, **kwargs):
        global _NB_PATH
        try:
            override = get_nb_override()
            new_path = override if override else guess_nb_path(repo_root)
            if new_path != "unknown.ipynb":
                _NB_PATH = new_path
                if not quiet:
                    print(f"[notebook_header] Detected notebook after first cell: {_NB_PATH}")
                try:
                    show_colab_button(org, repo, branch, _NB_PATH)
                except Exception:
                    pass
                try:
                    ip.events.unregister('post_run_cell', _callback)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            try:
                ip.events.unregister('post_run_cell', _callback)  # type: ignore[attr-defined]
            except Exception:
                pass

    try:
        ip.events.register('post_run_cell', _callback)  # type: ignore[attr-defined]
    except Exception:
        pass

# ---------------------------
# Main entry
# ---------------------------
def main(force: bool = False, quiet: bool = True):
    global _NB_PATH

    repo_root = get_repo_root()
    org, repo = get_org_repo()
    branch = os.environ.get("BOOT_BRANCH", DEFAULT_BOOT_BRANCH)

    if _STATE["ran"] and not force:
        # Re-show the badge so re-running the cell doesn't blank it out
        try:
            show_colab_button(org, repo, branch, _NB_PATH)
        except Exception as e:
            if not quiet:
                print("[notebook_header] Colab badge error:", e)
        return

    _STATE["ran"] = True
    # ... keep the rest of your function body the same, but remove the old early-return block ...

    repo_root = get_repo_root()
    org, repo = get_org_repo()
    branch = os.environ.get("BOOT_BRANCH", DEFAULT_BOOT_BRANCH)

    nb_override = get_nb_override()
    if nb_override:
        _NB_PATH = nb_override
        if not quiet:
            print(f"[notebook_header] Using NB_PATH override: {_NB_PATH}")
    elif repo_root:
        _NB_PATH = guess_nb_path(repo_root)
    else:
        _NB_PATH = "unknown.ipynb"

    if not quiet:
        print(f"[notebook_header] {org}/{repo}@{branch}")
        print(f"[notebook_header] Notebook: {_NB_PATH}")
        print(f"[notebook_header] Time: {datetime.datetime.now().isoformat(timespec='seconds')}")

    try:
        show_colab_button(org, repo, branch, _NB_PATH)
    except Exception as e:
        if not quiet:
            print("[notebook_header] Colab badge error:", e)

    if in_colab():
        try:
            colab_bootstrap(org, repo, branch)
        except Exception as e:
            if not quiet:
                print("[notebook_header] Colab bootstrap error:", e)

    if _NB_PATH == "unknown.ipynb" and in_ipython() and repo_root:
        _post_run_attempt_path(repo_root, org, repo, branch, quiet)

    try_auto_imports()

def reload_header(quiet: bool = True):
    return main(force=True, quiet=quiet)

# ---------------------------
# Autorun
# ---------------------------
if os.environ.get("NOTEBOOK_HEADER_AUTORUN", "1").lower() in ("1", "true", "yes"):
    if in_ipython():
        try:
            main(False)
        except Exception as e:
            print("[notebook_header] autorun failed:", e)