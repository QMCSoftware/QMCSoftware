import qmctoolscl 
import numpy as np 

def _parse_ft_input(x):
    assert x.ndim<=3
    if x.ndim==1: d1,d2,n = 1,1,x.shape[0]
    elif x.ndim==2: d1,d2,n = 1,*x.shape
    elif x.ndim==3: d1,d2,n = x.shape
    assert (n&(n-1))==0 # require n is 0 or a power of 2
    return d1,d2,n,n//2

def fft_bro_1d_radix2_py(x):
    d1,d2,n,n_half = _parse_ft_input(x)
    if n<=1: return x.copy()
    twiddler = np.empty(n,dtype=np.float64)
    twiddlei = np.empty(n,dtype=np.float64)
    xr = x.real.copy()
    xi = x.imag.copy()
    qmctoolscl.fft_bro_1d_radix2(d1,d2,int(n_half),twiddler,twiddlei,xr,xi)
    return xr+1j*xi

def ifft_bro_1d_radix2_py(x):
    d1,d2,n,n_half = _parse_ft_input(x)
    if n<=1: return x.copy()
    twiddler = np.empty(n,dtype=np.float64)
    twiddlei = np.empty(n,dtype=np.float64)
    xr = x.real.copy()
    xi = x.imag.copy()
    qmctoolscl.ifft_bro_1d_radix2(d1,d2,int(n_half),twiddler,twiddlei,xr,xi)
    return xr+1j*xi

def fwht_1d_radix2_py(x):
    d1,d2,n,n_half = _parse_ft_input(x)
    if n<=1: return x.copy()
    xcp = x.copy()
    qmctoolscl.fwht_1d_radix2(d1,d2,int(n_half),xcp)
    return xcp