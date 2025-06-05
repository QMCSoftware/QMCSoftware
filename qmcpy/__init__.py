from .discrete_distribution import *
from .true_measure import *
from .integrand import *
from .stopping_criterion import *
from .kernels import *
from .kernel_methods import (
    fftbr,ifftbr,fwht,
    KernelShiftInvar,KernelDigShiftInvar,KernelGaussian,
    FastGramMatrixLattice,FastGramMatrixDigitalNetB2,GramMatrix)
try:
    from .kernel_methods import (
        fftbr_torch,ifftbr_torch,fwht_torch
    )
except:
    pass
from .util import plot_proj

name = "qmcpy"
__version__ = "1.6.2"
