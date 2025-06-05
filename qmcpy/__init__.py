from .discrete_distribution import *
from .true_measure import *
from .integrand import *
from .stopping_criterion import *
from .util import kernel_methods
from .util.kernel_methods import (
    fftbr,ifftbr,fwht)
try:
    from .util.kernel_methods import (
        fftbr_torch,ifftbr_torch,fwht_torch
    )
except:
    pass
from .util import plot_proj

name = "qmcpy"
__version__ = "1.6.3b"
