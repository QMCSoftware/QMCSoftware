import torch 
import numpy as np 

SQRT2 = np.sqrt(2) 

def _fwht_torch(x):
    y = x.clone()
    n = x.size(-1)
    if n<=1: return y
    assert n&(n-1)==0 # require n is a power of 2
    m = int(np.log2(n))
    it = torch.arange(n,dtype=int).reshape([2]*m) # 2 x 2 x ... x 2 array (size 2^m)
    idx0 = [slice(None)]*(m-1)+[0]
    idx1 = [slice(None)]*(m-1)+[1]
    eps0 = [Ellipsis,None]
    eps1 = [Ellipsis,None]
    for k in range(m):
        eps0[1] = it[idx0[-(k+1):]].flatten()
        eps1[1] = it[idx1[-(k+1):]].flatten()
        y0,y1 = y[eps0],y[eps1]
        y[eps0],y[eps1] = (y0+y1)/SQRT2,(y0-y1)/SQRT2
    return y
class _FWHTB2Ortho(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        return _fwht_torch(x)
    @staticmethod
    def backward(self, dx):
        return _fwht_torch(dx)
def fwht_torch(x):
    """
    >>> rng = np.random.Generator(np.random.SFC64(11))
    >>> x = torch.from_numpy(rng.random(8)).float().requires_grad_()
    >>> y = fwht_torch(x)
    >>> v = torch.sum(y**2)
    >>> dvdx = torch.autograd.grad(v,x)[0]
    >>> x.detach()
    tensor([0.2516, 0.7325, 0.0566, 0.6132, 0.4502, 0.2550, 0.3491, 0.3175])
    >>> y.detach()
    tensor([ 1.0697, -0.2866,  0.1248,  0.0846,  0.0997, -0.4470,  0.0975, -0.0311])
    >>> v.detach().item()
    1.469...
    >>> dvdx.detach()
    tensor([0.5032, 1.4650, 0.1132, 1.2263, 0.9003, 0.5100, 0.6982, 0.6350])
    """
    return _FWHTB2Ortho.apply(x)

def _bitrev(x, m):
    r = 0
    for k in range(m):
        r += (x&1)<<(m-k-1)
        x >>=1
    return r 

def fftbr_torch(x):
    """
    >>> rng = np.random.Generator(np.random.SFC64(11))
    >>> x = torch.from_numpy(rng.random(8)+1j*rng.random(8)).to(torch.complex64).requires_grad_()
    >>> y = fftbr_torch(x)
    >>> v = torch.abs(torch.sum(y**2))
    >>> dvdx = torch.autograd.grad(v,x)[0]
    >>> x.detach()
    tensor([0.2516+0.6480j, 0.7325+0.5961j, 0.0566+0.2046j, 0.6132+0.3853j,
            0.4502+0.0554j, 0.2550+0.0930j, 0.3491+0.3887j, 0.3175+0.8507j])
    >>> y.detach()
    tensor([ 1.0697+1.1391j, -0.3179+0.2646j, -0.2746+0.2177j, -0.2719-0.3412j,
             0.0997+0.1577j, -0.1499+0.1657j,  0.4968+0.2449j,  0.0596-0.0156j])
    >>> v.detach().item()
    2.558...
    >>> dvdx.detach()
    tensor([1.3026+0.4861j, 1.2113+1.4492j, 0.7866+1.2161j, 0.4107+0.1078j,
            1.7096+0.6126j, 0.7865+0.6879j, 0.1926+0.5075j, 0.1227+0.8988j])
    """
    n = x.size(-1)
    assert n&(n-1)==0 # require n is a power of 2
    m = int(np.log2(n))
    return torch.fft.fft(x[...,_bitrev(torch.arange(n,device=x.device),m)],norm="ortho")

def ifftbr_torch(x):
    """
    >>> rng = np.random.Generator(np.random.SFC64(11))
    >>> x = torch.from_numpy(rng.random(8)+1j*rng.random(8)).to(torch.complex64).requires_grad_()
    >>> y = ifftbr_torch(x)
    >>> v = torch.abs(torch.sum(y**2))
    >>> dvdx = torch.autograd.grad(v,x)[0]
    >>> x.detach()
    tensor([0.2516+0.6480j, 0.7325+0.5961j, 0.0566+0.2046j, 0.6132+0.3853j,
            0.4502+0.0554j, 0.2550+0.0930j, 0.3491+0.3887j, 0.3175+0.8507j])
    >>> y.detach()
    tensor([ 1.0697+1.1391j, -0.2866-0.2221j,  0.2980+0.0590j, -0.0887+0.0189j,
             0.0309+0.5415j, -0.0412-0.3293j, -0.1902+0.2641j, -0.0804+0.3618j])
    >>> v.detach().item()
    2.565...
    >>> dvdx.detach()
    tensor([ 1.1474+0.7852j,  1.5120+1.0060j,  0.5979+0.8570j,  0.0649+0.5390j,
            -0.0972+0.9019j,  0.4709+1.3696j,  0.3727+0.2035j,  0.8270+1.6981j])
    """
    n = x.size(-1)
    assert n&(n-1)==0 # require n is a power of 2
    m = int(np.log2(n))
    return torch.fft.ifft(x,norm="ortho")[...,_bitrev(torch.arange(n,device=x.device),m)]
