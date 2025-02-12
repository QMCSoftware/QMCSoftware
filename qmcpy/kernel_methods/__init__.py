from .util import (
    bernoulli_poly,
    weighted_walsh_funcs)
from .pcg_module import pcg,IdentityPrecond,PPCholPrecond,JacobiPrecond,SSORPrecond,BlockPrecond

from .fast_transforms import (
    fftbr,ifftbr,fwht
)
try:
    from .fast_transforms import (
        fftbr_torch,ifftbr_torch,fwht_torch
    )
except:
    pass
from .kernel import (
    KernelGaussian,
    KernelShiftInvar,KernelDigShiftInvar)
from .gram_matrix import (
    GramMatrix,
    FastGramMatrixLattice,FastGramMatrixDigitalNetB2)
from .pde_gram_matrix import (
    PDEGramMatrix,
    FastPDEGramMatrix)
from .gpr import GPR,FGPRLattice,FGPRDigitalNetB2
