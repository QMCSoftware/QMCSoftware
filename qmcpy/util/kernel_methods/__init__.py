from .ft_qmctoolscl import fftbr,ifftbr,fwht,omega_fftbr,omega_fwht
from .shift_invar_ops import bernoulli_poly, kernel_shift_invar
from .dig_shift_invar_ops import weighted_walsh_funcs, kernel_dig_shift_invar, bin_to_float, float_to_bin, bin_from_numpy_to_float
kernel_si = kernel_shift_invar
kernel_dsi = kernel_dig_shift_invar
try:
    import torch 
    from .ft_pytorch import (
        fftbr_torch,
        ifftbr_torch,
        fwht_torch,
        omega_fftbr_torch,
        omega_fwht_torch,
    )
except ImportError:
    def fftbr_torch(*args, **kwargs):
        raise Exception("fftbr_torch requires torch but no installation found")
    def ifftbr_torch(*args, **kwargs):
        raise Exception("ifftbr_torch requires torch but no installation found") 
    def fwht_torch(*args, **kwargs):
        raise Exception("fwht_torch requires torch but no installation found") 
    def omega_fftbr_torch(*args, **kwargs):
        raise Exception("omega_fftbr_torch requires torch but no installation found") 
    def omega_fwht_torch(*args, **kwargs):
        raise Exception("omega_fwht_torch requires torch but no installation found") 

