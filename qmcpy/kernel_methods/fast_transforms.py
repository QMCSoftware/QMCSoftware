import qmctoolscl 
import numpy as np 

def _parse_ft_input(x):
    assert x.ndim<=3
    if x.ndim==1: d1,d2,n = 1,1,x.shape[0]
    elif x.ndim==2: d1,d2,n = 1,*x.shape
    elif x.ndim==3: d1,d2,n = x.shape
    assert (n&(n-1))==0 # require n is 0 or a power of 2
    return d1,d2,n,n//2

def fft_bro_1d_radix2(x):
    """
    1 dimensional Bit-Reversed-Order (BRO) Fast Fourier Transform (FFT) along the last dimension. 
    Requires the last dimension of x is already in BRO, so we can skip the first step of the decimation-in-time FFT. 
    Requires the size of the last dimension is a power of 2. 
    Requires number of dimension of the input is less than 3. 

    Args:
        x (ndarray): array of samples at which to run BRO-FFT. Requires x.ndim<=3
    
    Returns:
        ndarray: BRO-FFT values
    """
    d1,d2,n,n_half = _parse_ft_input(x)
    if n<=1: return x.copy()
    twiddler = np.empty(n,dtype=np.float64)
    twiddlei = np.empty(n,dtype=np.float64)
    xr = x.real.copy()
    xi = x.imag.copy()
    qmctoolscl.fft_bro_1d_radix2(d1,d2,int(n_half),twiddler,twiddlei,xr,xi)
    return xr+1j*xi

def ifft_bro_1d_radix2(x):
    """
    1 dimensional Bit-Reversed-Order (BRO) Inverse Fast Fourier Transform (IFFT) along the last dimension. 
    Outputs  an array in bit-reversed order, so we can skip the last step of the decimation-in-time IFFT. 
    Requires the size of the last dimension is a power of 2. 
    Requires number of dimension of the input is less than 3. 

    Args:
        x (ndarray): array of samples at which to run BRO-IFFT. Requires x.ndim<=3
    
    Returns:
        ndarray: BRO-IFFT values
    """
    d1,d2,n,n_half = _parse_ft_input(x)
    if n<=1: return x.copy()
    twiddler = np.empty(n,dtype=np.float64)
    twiddlei = np.empty(n,dtype=np.float64)
    xr = x.real.copy()
    xi = x.imag.copy()
    qmctoolscl.ifft_bro_1d_radix2(d1,d2,int(n_half),twiddler,twiddlei,xr,xi)
    return xr+1j*xi

def fwht_1d_radix2(x):
    """
    1 dimensional Fast Walsh Hadamard Transform (FWHT) . 
    Requires the size of the last dimension is a power of 2. 
    Requires number of dimension of the input is less than 3. 

    Args:
        x (ndarray): array of samples at which to run FWHT. Requires x.ndim<=3
    
    Returns:
        ndarray: FWHT values
    """
    d1,d2,n,n_half = _parse_ft_input(x)
    if n<=1: return x.copy()
    xcp = x.copy()
    qmctoolscl.fwht_1d_radix2(d1,d2,int(n_half),xcp)
    return xcp

