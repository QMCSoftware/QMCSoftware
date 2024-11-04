import qmctoolscl 
import numpy as np 

def _parse_ft_input(x):
    if x.ndim==1:
        d,n = 1,len(x)
        x = x[None,:]
        shape = (n,)
    elif x.ndim==2:
        d,n = x.shape
        shape = (d,n)
    else:
        shape = x.shape
        n = shape[-1]
        x = x.reshape(-1,n)
        d = x.shape[0]
    assert (n&(n-1))==0 # require n is 0 or a power of 2
    return x,shape,d,n,n//2

def fft_bro_1d_radix2(x):
    """
    1 dimensional Bit-Reversed-Order (BRO) Fast Fourier Transform (FFT) along the last dimension. 
    Requires the last dimension of x is already in BRO, so we can skip the first step of the decimation-in-time FFT. 
    Requires the size of the last dimension is a power of 2. 

    Args:
        x (ndarray): array of samples at which to run BRO-FFT. Requires x.ndim<=3
    
    Returns:
        ndarray: BRO-FFT values
    """
    x,shape,d,n,n_half = _parse_ft_input(x)
    if n<=1: return x.reshape(shape).copy()
    twiddler = np.empty(n,dtype=np.float64)
    twiddlei = np.empty(n,dtype=np.float64)
    xr = x.real.copy()
    xi = x.imag.copy()
    qmctoolscl.fft_bro_1d_radix2(1,d,int(n_half),twiddler,twiddlei,xr,xi)
    xc = xr+1j*xi
    return xc.reshape(shape)

def ifft_bro_1d_radix2(x):
    """
    1 dimensional Bit-Reversed-Order (BRO) Inverse Fast Fourier Transform (IFFT) along the last dimension. 
    Outputs  an array in bit-reversed order, so we can skip the last step of the decimation-in-time IFFT. 
    Requires the size of the last dimension is a power of 2. 

    Args:
        x (ndarray): array of samples at which to run BRO-IFFT. Requires x.ndim<=3
    
    Returns:
        ndarray: BRO-IFFT values
    """
    x,shape,d,n,n_half = _parse_ft_input(x)
    if n<=1: return x.reshape(shape).copy()
    twiddler = np.empty(n,dtype=np.float64)
    twiddlei = np.empty(n,dtype=np.float64)
    xr = x.real.copy()
    xi = x.imag.copy()
    qmctoolscl.ifft_bro_1d_radix2(1,d,int(n_half),twiddler,twiddlei,xr,xi)
    xc = xr+1j*xi
    return xc.reshape(shape)


def fwht_1d_radix2(x):
    """
    1 dimensional Fast Walsh Hadamard Transform (FWHT) . 
    Requires the size of the last dimension is a power of 2. 

    Args:
        x (ndarray): array of samples at which to run FWHT. Requires x.ndim<=3
    
    Returns:
        ndarray: FWHT values
    """
    x,shape,d,n,n_half = _parse_ft_input(x)
    if n<=1: return x.reshape(shape).copy()
    xcp = x.copy()
    qmctoolscl.fwht_1d_radix2(1,d,int(n_half),xcp)
    return xcp.reshape(shape)

