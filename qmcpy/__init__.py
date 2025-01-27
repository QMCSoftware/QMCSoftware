from .discrete_distribution import *
from .true_measure import *
from .integrand import *
from .stopping_criterion import *
from .util import plot_proj 
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources

import subprocess
import sys

def qmc_apply_style():
    """Apply the qmcpy matplotlib style to allow for a standard style."""
    style_file = 'qmcpy.mplstyle'

    try:
        with pkg_resources.path(__package__, style_file) as style_path:
            plt.style.use(style_path)
        print("qmcpy matplotlib style applied.")
    except FileNotFoundError:
        print(f"Style file {style_file} not found in package {__package__}.")

name = "qmcpy"
__version__ = "1.5"
