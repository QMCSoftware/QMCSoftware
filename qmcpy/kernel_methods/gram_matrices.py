from ..discrete_distribution import Lattice,DigitalNetB2
from .kernels import KernelShiftInvar,KernelDigShiftInvar
from .fast_transforms import fft_bro_1d_radix2,ifft_bro_1d_radix2,fwht_1d_radix2
import numpy as np
    
class FastGramMatrix(object):
    def __init__(self, dd_obj, kernel_obj, n, ft, ift):
        assert dd_obj.d==kernel_obj.d 
        assert dd_obj.replications==1
        assert (n&(n-1))==0 # require n is 0 or a power of 2
        self.ft = ft 
        self.ift = ift

class FastGramMatrixLattice(FastGramMatrix):
    """
    Fast Gram matrix operations using lattice points and shift invariant kernels 
    
    >>> n = 2**3 
    >>> d = 3 
    >>> lat_obj = Lattice(d,randomize="SHIFT",seed=7)
    >>> kernel_si = KernelShiftInvar(d)
    >>> gm = FastGramMatrixLattice(lat_obj,kernel_si,n)
    """
    def __init__(self, lat_obj, kernel_si, n):
        assert isinstance(lat_obj,Lattice) and lat_obj.randomize=="SHIFT"
        assert isinstance(kernel_si,KernelShiftInvar)
        super(FastGramMatrixLattice,self).__init__(lat_obj,kernel_si,n,fft_bro_1d_radix2,ifft_bro_1d_radix2)
        self.x = lat_obj.gen_samples(n_min=0,n_max=n)

class FastGramMatrixDigitalNetB2(FastGramMatrix):
    """
    Fast Gram matrix operations using base 2 digital net points and digitally shift invariant kernels 
    
    >>> n = 2**3 
    >>> d = 3 
    >>> dnb2_obj = DigitalNetB2(d,randomize="LMS_DS",seed=7)
    >>> kernel_dsi = KernelDigShiftInvar(d)
    >>> gm = FastGramMatrixDigitalNetB2(dnb2_obj,kernel_dsi,n)
    """
    def __init__(self, dnb2_obj, kernel_dsi, n):
        assert isinstance(dnb2_obj,DigitalNetB2) and dnb2_obj.randomize=="LMS_DS"
        assert isinstance(kernel_dsi,KernelDigShiftInvar)
        kernel_dsi.set_t(dnb2_obj.t_lms)
        super(FastGramMatrixDigitalNetB2,self).__init__(dnb2_obj,kernel_dsi,n,fwht_1d_radix2,fwht_1d_radix2)
        self.x = dnb2_obj.gen_samples(n_min=0,n_max=n,return_binary=True)
    
