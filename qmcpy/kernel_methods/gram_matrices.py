from ..discrete_distribution import Lattice,DigitalNetB2
from .kernels import KernelShiftInvar,KernelDigShiftInvar
from .fast_transforms import fft_bro_1d_radix2,ifft_bro_1d_radix2,fwht_1d_radix2
import numpy as np

class _FastGramMatrix(object):
    def __init__(self, dd_obj, kernel_obj, n, ft, ift, noise, dd_type, dd_randomize_ops, dd_order, kernel_type_ops):
        self.dd_obj = dd_obj
        self.kernel_obj = kernel_obj
        self.n = n
        self.ft = ft 
        self.ift = ift
        noise = noise
        assert self.dd_obj.d==self.kernel_obj.d 
        assert self.dd_obj.replications==1
        assert (self.n&(self.n-1))==0 # require n is 0 or a power of 2
        assert isinstance(noise,float) and noise>=0
        assert isinstance(dd_obj,dd_type) and dd_obj.randomize in dd_randomize_ops and dd_obj.order==dd_order
        assert any(isinstance(kernel_obj,kernel_type_op) for kernel_type_op in kernel_type_ops)
        self._x,self.x = self.sample(0,n)
        k1 = self.kernel_obj(self._x,self._x[[0]])[:,0]
        self.lam = np.sqrt(self.n)*self.ft(k1)
        self.lam[0] = self.lam[0]+noise
    def get_full_gram_matrix(self):
        return self.kernel_obj(self._x,self._x)
    def multiply_yt(self, yt):
        ytogndim = yt.ndim
        assert ytogndim<=2 
        if ytogndim==1: yt = yt[:,None]
        assert yt.ndim==2 and yt.shape[0]==self.n
        sol = self.ift(yt.T*self.lam).real.T
        return sol[:,0] if ytogndim==1 else sol
    def multiply(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None]
        assert y.ndim==2 and y.shape[0]==self.n
        yt = self.ft(y.T).T
        sol = self.multiply_yt(yt)
        return sol[:,0] if yogndim==1 else sol
    def __matmul__(self, y):
        return self.multiply(y)
    def solve_yt(self, yt):
        ytogndim = yt.ndim
        assert ytogndim<=2 
        if ytogndim==1: yt = yt[:,None]
        assert yt.ndim==2 and yt.shape[0]==self.n
        sol = self.ift(yt.T/self.lam).real.T
        return sol[:,0] if ytogndim==1 else sol
    def solve(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None]
        assert y.ndim==2 and y.shape[0]==self.n
        yt = self.ft(y.T).T
        sol = self.solve_yt(yt)
        return sol[:,0] if yogndim==1 else sol

class FastGramMatrixLattice(_FastGramMatrix):
    """
    Fast Gram matrix operations using lattice points and shift invariant kernels 
    
    >>> n = 2**3 
    >>> d = 3 
    >>> lat_obj = Lattice(d,seed=7)
    >>> kernel_si = KernelShiftInvar(d)
    >>> gm = FastGramMatrixLattice(lat_obj,kernel_si,n)
    >>> y = np.vstack([np.sin(2*np.pi*gm.x).prod(1),np.cos(2*np.pi*gm.x).prod(1)]).T
    >>> y.shape
    (8, 2)
    >>> gmatfull = gm.get_full_gram_matrix()
    >>> assert np.allclose(gm@y[:,0],gmatfull@y[:,0],atol=1e-12)
    >>> assert np.allclose(gm@y,gmatfull@y,atol=1e-12)
    >>> assert np.allclose(gm.solve(y[:,0]),np.linalg.solve(gmatfull,y[:,0]),atol=1e-12)
    >>> assert np.allclose(gm.solve(y),np.linalg.solve(gmatfull,y),atol=1e-12)
    >>> yt = gm.ft(y.T).T
    >>> yt.shape 
    (8, 2)
    >>> assert np.allclose(gm.multiply_yt(yt[:,0]),gmatfull@y[:,0],atol=1e-12)
    >>> assert np.allclose(gm.multiply_yt(yt),gmatfull@y,atol=1e-12)
    >>> assert np.allclose(gm.solve_yt(yt[:,0]),np.linalg.solve(gmatfull,y[:,0]),atol=1e-12)
    >>> assert np.allclose(gm.solve_yt(yt),np.linalg.solve(gmatfull,y),atol=1e-12)
    """
    def __init__(self, lat_obj, kernel_si, n, noise=0.):
        """
        Args:
            lat_obj (Lattice): lattice discrete distribution with randomize='shift'
            kernel_si (KernelShiftInvar): shift invariant kernel
            n (int): power of two number of points
            noise (float): nugget term 
        """
        super(FastGramMatrixLattice,self).__init__(lat_obj,kernel_si,n,fft_bro_1d_radix2,ifft_bro_1d_radix2,noise,
            dd_type = Lattice, 
            dd_randomize_ops = ["SHIFT"],
            dd_order = "NATURAL",
            kernel_type_ops = [KernelShiftInvar])
    def sample(self, n_min, n_max):
        x = self.dd_obj.gen_samples(n_min=n_min,n_max=n_max)
        return x,x

class FastGramMatrixDigitalNetB2(_FastGramMatrix):
    """
    Fast Gram matrix operations using base 2 digital net points and digitally shift invariant kernels 
    
    >>> n = 2**3 
    >>> d = 3 
    >>> dnb2_obj = DigitalNetB2(d,randomize="LMS_DS",seed=7)
    >>> kernel_dsi = KernelDigShiftInvar(d)
    >>> gm = FastGramMatrixDigitalNetB2(dnb2_obj,kernel_dsi,n)
    >>> y = np.vstack([np.sin(2*np.pi*gm.x).prod(1),np.cos(2*np.pi*gm.x).prod(1)]).T
    >>> y.shape
    (8, 2)
    >>> gmatfull = gm.get_full_gram_matrix()
    >>> assert np.allclose(gm@y[:,0],gmatfull@y[:,0],atol=1e-12)
    >>> assert np.allclose(gm@y,gmatfull@y,atol=1e-12)
    >>> assert np.allclose(gm.solve(y[:,0]),np.linalg.solve(gmatfull,y[:,0]),atol=1e-12)
    >>> assert np.allclose(gm.solve(y),np.linalg.solve(gmatfull,y),atol=1e-12)
    >>> yt = gm.ft(y.T).T
    >>> yt.shape 
    (8, 2)
    >>> assert np.allclose(gm.multiply_yt(yt[:,0]),gmatfull@y[:,0],atol=1e-12)
    >>> assert np.allclose(gm.multiply_yt(yt),gmatfull@y,atol=1e-12)
    >>> assert np.allclose(gm.solve_yt(yt[:,0]),np.linalg.solve(gmatfull,y[:,0]),atol=1e-12)
    >>> assert np.allclose(gm.solve_yt(yt),np.linalg.solve(gmatfull,y),atol=1e-12)
    """
    def __init__(self, dnb2_obj, kernel_dsi, n, noise=0.):
        """
        Args:
            lat_obj (Lattice): lattice discrete distribution with randomize='shift'
            kernel_si (KernelShiftInvar): shift invariant kernel
            n (int): power of two number of points
            noise (float): nugget term 
        """
        kernel_dsi.set_t(dnb2_obj.t_lms)
        super(FastGramMatrixDigitalNetB2,self).__init__(dnb2_obj,kernel_dsi,n,fwht_1d_radix2,fwht_1d_radix2,noise,
            dd_type = DigitalNetB2, 
            dd_randomize_ops = ["DS","LMS_DS"],
            dd_order = "NATURAL",
            kernel_type_ops = [KernelDigShiftInvar])
    def sample(self, n_min, n_max):
        xb = self.dd_obj.gen_samples(n_min=n_min,n_max=n_max,return_binary=True)
        xf = xb*2**(-self.kernel_obj.t)
        return xb,xf

    
