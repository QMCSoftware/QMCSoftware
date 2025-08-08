# utils/colab_button.py
import os, sys
from typing import Optional
from IPython.display import HTML, display

def show(org: str, repo: str, branch: str, nb_path: str, message: Optional[str] = None):
    """Display an 'Open in Colab' button conditionally.

    - Hidden if already in Colab or in a conda env.
    - `nb_path` is the path to *this* notebook within the repo.
    """
    IN_COLAB = "google.colab" in sys.modules
    IN_CONDA = bool(os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV"))
    if IN_COLAB or IN_CONDA:
        return
    url = f"https://colab.research.google.com/github/{org}/{repo}/blob/{branch}/{nb_path}"
    msg = message or "Not in a conda env — click to run this notebook on Colab."
    display(HTML(f'''
    <div style="margin:1em 0">
      <a href="{url}" target="_blank" rel="noopener">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
      </a>
      <div style="font-size:0.9em;color:#666;margin-top:0.3em">{msg}</div>
    </div>'''))