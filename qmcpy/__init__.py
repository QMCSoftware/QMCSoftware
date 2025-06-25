from .discrete_distribution import *
from .true_measure import *
from .integrand import *
from .stopping_criterion import *
from .util import kernel_methods
from .util.kernel_methods import fftbr,ifftbr,fwht,omega_fftbr,omega_fwht
from .util.kernel_methods import fftbr_torch,ifftbr_torch,fwht_torch,omega_fftbr_torch,omega_fwht_torch
from .util import plot_proj

name = "qmcpy"
__version__ = "1.6.3c"
