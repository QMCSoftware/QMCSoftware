import os
import sys
from IPython.display import HTML, display
from typing import Optional

# Single point of truth for branch — change default after merge
DEFAULT_BRANCH = os.environ.get("BOOT_BRANCH", "bootstrap_colab")

def show(
    org: str,
    repo: str,
    nb_path: str,
    *,
    branch: Optional[str] = None,
    message: Optional[str] = None,
    new_tab: bool = True,
):
    """
    Display an 'Open in Colab' button if NOT already in Colab or a conda env.

    Args:
        org: GitHub org/user name.
        repo: GitHub repo name.
        nb_path: Path to the notebook inside the repo.
        branch: Optional override for the Git branch (defaults to DEFAULT_BRANCH).
        message: Optional text under the badge.
        new_tab: If True, opens in new tab (avoids replacing current notebook).
    """
    IN_COLAB = "google.colab" in sys.modules
    IN_CONDA = bool(os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV"))
    if IN_COLAB or IN_CONDA:
        return
    effective_branch = branch or DEFAULT_BRANCH
    url = f"https://colab.research.google.com/github/{org}/{repo}/blob/{effective_branch}/{nb_path}"
    msg = message or "Not in a conda env — click to run this notebook on Colab."
    target = ' target="_blank" rel="noopener"' if new_tab else ""
    display(HTML(f'''
    <div style="margin:1em 0">
      <a href="{url}"{target}>
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
      </a>
      <div style="font-size:0.9em;color:#666;margin-top:0.3em">{msg}</div>
    </div>'''))