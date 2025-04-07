from .ft_qmctoolscl import fftbr,ifftbr,fwht 
try:
    import torch 
    from .ft_pytorch import fftbr_torch,ifftbr_torch,fwht_torch
except:
    pass 
