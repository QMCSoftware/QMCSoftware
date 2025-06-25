from .kernel_methods import *
from .exceptions_warnings import *
from .abstraction_functions import _univ_repr
from .latnetbuilder_linker import latnetbuilder_linker
from .plot_functions import plot_proj
from .stop_notebook import stop_notebook
try: 
    import torch 
    import gpytorch
    from .exact_gpytorch_gression_model import ExactGPyTorchRegressionModel
except ImportError:
    class ExactGPyTorchRegressionModel(object):
        def __init__(self, *args, **kwargs):
            raise Exception("ExactGPyTorchRegressionModel requires torch and gpytorch but no installations found")

