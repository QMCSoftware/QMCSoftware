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

def _fftbr_torch(x):
    n = x.size(-1)
    assert n&(n-1)==0 # require n is a power of 2
    m = int(np.log2(n))
    y = x.clone()+0j
    if n<=1: return y
    it = torch.arange(n,dtype=int).reshape([2]*m) # 2 x 2 x ... x 2 array (size 2^m)
    t0 = torch.tensor(0)
    t1 = torch.tensor(1)
    twiddle = torch.exp(-2*np.pi*1j*torch.arange(n,device=y.device)/n)
    for k in range(m):
        s = m-k-1
        f = 1<<k 
        i1v = torch.index_select(it,dim=s,index=t0).flatten()
        i2v = torch.index_select(it,dim=s,index=t1).flatten()
        y1,y2 = y[...,i1v],y[...,i2v]
        t = (i1v%f)*(1<<s)
        z = twiddle[t]*y2 
        y[...,i1v],y[...,i2v] = (y1+z)/SQRT2,(y1-z)/SQRT2
    return y
class _FFTB2OrthoBRO(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        return _fftbr_torch(x)
    @staticmethod
    def backward(self, dx):
        return _fftbr_torch(dx)
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
    tensor([1.3026+0.4861j, 0.8304+0.3550j, 0.6621+0.8429j, 1.1155+1.3293j,
            0.5114-0.2196j, 0.4179+0.8007j, 1.5217+0.9647j, 0.1611+1.4069j])
    """
    return _FFTB2OrthoBRO.apply(x)

def _ifftbr_torch(x):
    y = x.clone().to(torch.complex64)
    n = x.size(-1)
    if n<=1: return y
    assert n&(n-1)==0 # require n is a power of 2
    m = int(np.log2(n))
    it = torch.arange(n,dtype=int).reshape([2]*m) # 2 x 2 x ... x 2 array (size 2^m)
    t0 = torch.tensor(0)
    t1 = torch.tensor(1)
    twiddle = torch.exp(2*np.pi*1j*torch.arange(n,device=y.device)/n)
    for k in range(m):
        s = m-k-1
        i1v = torch.index_select(it,dim=k,index=t0).flatten()
        i2v = torch.index_select(it,dim=k,index=t1).flatten()
        y1,y2 = y[...,i1v],y[...,i2v]
        t = (i1v%(1<<s))*(1<<k)
        y[...,i1v],y[...,i2v] = (y1+y2)/SQRT2,twiddle[t]*(y1-y2)/SQRT2
    return y
class _IFFTB2OrthoBRO(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        return _ifftbr_torch(x)
    @staticmethod
    def backward(self, dx):
        return _ifftbr_torch(dx)
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
    tensor([ 1.1474+0.7852j,  1.2224+1.5245j,  1.0341+0.2861j,  0.5153+1.5184j,
             0.3375+0.4416j, -0.4556+0.7417j,  0.4514+1.0096j,  0.6429+1.0530j])
    """
    return _IFFTB2OrthoBRO.apply(x)
