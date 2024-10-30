from ..discrete_distribution import Lattice,DigitalNetB2,DiscreteDistribution
from .kernels import KernelShiftInvar,KernelDigShiftInvar
from .fast_transforms import fft_bro_1d_radix2,ifft_bro_1d_radix2,fwht_1d_radix2
import numpy as np

def _mult_check(gm):
    y = np.vstack([np.sin(2*np.pi*gm.x).prod(1),np.cos(2*np.pi*gm.x).prod(1)]).T
    gmatfull = gm.get_full_gram_matrix()
    assert np.allclose(gm@y[:,0],gmatfull@y[:,0],atol=1e-12)
    assert np.allclose(gm@y,gmatfull@y,atol=1e-12)
    yt = gm.ft(y.T).T
    assert np.allclose(gm.multiply_yt(yt[:,0]),gmatfull@y[:,0],atol=1e-12)
    assert np.allclose(gm.multiply_yt(yt),gmatfull@y,atol=1e-12)

def _solve_check(gm):
    y = np.vstack([np.sin(2*np.pi*gm.x).prod(1),np.cos(2*np.pi*gm.x).prod(1)]).T
    gmatfull = gm.get_full_gram_matrix()
    assert np.allclose(gm.solve(y[:,0]),np.linalg.solve(gmatfull,y[:,0]),atol=1e-12)
    assert np.allclose(gm.solve(y),np.linalg.solve(gmatfull,y),atol=1e-12)
    yt = gm.ft(y.T).T
    assert np.allclose(gm.solve_yt(yt[:,0]),np.linalg.solve(gmatfull,y[:,0]),atol=1e-12)
    assert np.allclose(gm.solve_yt(yt),np.linalg.solve(gmatfull,y),atol=1e-12)
    
class _FastGramMatrix(object):
    """
    >>> n = 2**3 
    >>> d = 3 
    >>> gm_lat = FastGramMatrixLattice(Lattice(d,seed=7),KernelShiftInvar(d),n,n)
    >>> gm_dnb2 = FastGramMatrixDigitalNetB2(DigitalNetB2(d,seed=7),KernelDigShiftInvar(d),n,n)
    >>> for gm in [gm_lat,gm_dnb2]:
    ...     _mult_check(gm)
    ...     _solve_check(gm)     
    """
    def __init__(self, dd_obj_or__x_x, kernel_obj, n1, n2, noise, ft, ift, dd_type, dd_randomize_ops, dd_order, kernel_type_ops):
        self.kernel_obj = kernel_obj
        self.ft = ft 
        self.ift = ift
        self.n1 = n1 
        self.n2 = n2 
        assert (self.n1&(self.n1-1))==0 and (self.n2&(self.n2-1))==0 and self.n1>0 and self.n2>0 # require n1 and n2 are powers of 2
        #self.noise = noise
        assert noise==0
        #assert isinstance(self.noise,float) and self.noise>=0
        if isinstance(dd_obj_or__x_x,DiscreteDistribution):
            self.dd_obj = dd_obj_or__x_x
            assert isinstance(self.dd_obj,dd_type) and self.dd_obj.d==self.kernel_obj.d 
            assert self.dd_obj.replications==1
            assert self.dd_obj.randomize in dd_randomize_ops and self.dd_obj.order==dd_order
            self.nmax = max(n1,n2)
            self._x,self.x = self.sample(0,self.nmax)
        else:
            assert isinstance(dd_obj_or__x_x,tuple) and len(dd_obj_or__x_x)==2
            self._x,self.x = dd_obj_or__x_x
            self.n_max = len(self._x)
            assert self.n_max>=self.n1 and self.n_max>=self.n2
            assert self._x.shape==(self.n_max,self.kernel_obj.d) and self.x.shape==(self.n_max,self.kernel_obj.d)
        assert any(isinstance(kernel_obj,kernel_type_op) for kernel_type_op in kernel_type_ops)
        if self.n1==self.n2:
            self.vhs = "square"
            k1 = self.kernel_obj(self._x[:n1,:],self._x[[0],:])[:,0]
            self.lam = np.sqrt(self.n1)*self.ft(k1)
        elif self.n1>self.n2:
            self.vhs = "tall"
            self.rat = self.n1//self.n2
            k1 = self.kernel_obj(self._x[:n1,:],self._x[[0],:])[:,0].reshape((self.rat,self.n2))
            self.lam = np.sqrt(self.n2)*self.ft(k1)
        else: # self.n1<self.n2
            self.vhs = "wide"
            self.rat = self.n2//self.n1
            k1 = self.kernel_obj(self._x[[[0]],:],self._x[:n2,:])[0,:].reshape((self.rat,self.n1))
            self.lam = np.sqrt(self.n1)*self.ft(k1)
    def sample(self, n_min, n_max):
        assert hasattr(self,"dd_obj"), "no discrete distribution object available to sample from"
        _x,x = self._sample(n_min,n_max)
        return _x,x
    def get_full_gram_matrix(self):
        return self.kernel_obj(self._x,self._x)
    def multiply_yt(self, yt):
        ytogndim = yt.ndim
        assert ytogndim<=2 
        if ytogndim==1: yt = yt[:,None]
        assert yt.ndim==2 and yt.shape[0]==self.n2
        sol = self.ift(yt.T*self.lam).real.T
        return sol[:,0] if ytogndim==1 else sol
    def multiply(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None]
        assert y.ndim==2 and y.shape[0]==self.n2
        yt = self.ft(y.T).T
        sol = self.multiply_yt(yt)
        return sol[:,0] if yogndim==1 else sol
    def __matmul__(self, y):
        return self.multiply(y)
    def solve_yt(self, yt):
        assert self.vhs=="square", "cannot solve system in non-square matrix"
        ytogndim = yt.ndim
        assert ytogndim<=2 
        if ytogndim==1: yt = yt[:,None]
        assert yt.ndim==2 and yt.shape[0]==self.n1
        sol = self.ift(yt.T/self.lam).real.T
        return sol[:,0] if ytogndim==1 else sol
    def solve(self, y):
        assert self.vhs=="square", "cannot solve system in non-square matrix"
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None]
        assert y.ndim==2 and y.shape[0]==self.n1
        yt = self.ft(y.T).T
        sol = self.solve_yt(yt)
        return sol[:,0] if yogndim==1 else sol

class FastGramMatrixLattice(_FastGramMatrix):
    """
    Fast Gram matrix operations using lattice points and shift invariant kernels 
    """
    def __init__(self, lat_obj, kernel_si, n1, n2, noise=0.):
        """
        Args:
            lat_obj (Lattice): lattice discrete distribution with randomize='shift'
            kernel_si (KernelShiftInvar): shift invariant kernel
            n1 (int): first number of points
            n2 (int): second number of points
            noise (float): nugget term 
        """
        super(FastGramMatrixLattice,self).__init__(lat_obj,kernel_si,n1,n2,noise,
            ft = fft_bro_1d_radix2,
            ift = ifft_bro_1d_radix2,
            dd_type = Lattice, 
            dd_randomize_ops = ["SHIFT"],
            dd_order = "NATURAL",
            kernel_type_ops = [KernelShiftInvar])
    def _sample(self, n_min, n_max):
        x = self.dd_obj.gen_samples(n_min=n_min,n_max=n_max)
        return x,x

class FastGramMatrixDigitalNetB2(_FastGramMatrix):
    """
    Fast Gram matrix operations using base 2 digital net points and digitally shift invariant kernels 
    """
    def __init__(self, dnb2_obj, kernel_dsi, n1, n2, noise=0.):
        """
        Args:
            lat_obj (Lattice): lattice discrete distribution with randomize='shift'
            kernel_si (KernelShiftInvar): shift invariant kernel
             n1 (int): first number of points
            n2 (int): second number of points
            noise (float): nugget term 
        """
        kernel_dsi.set_t(dnb2_obj.t_lms)
        super(FastGramMatrixDigitalNetB2,self).__init__(dnb2_obj,kernel_dsi,n1,n2,noise,
            ft = fwht_1d_radix2,
            ift = fwht_1d_radix2,
            dd_type = DigitalNetB2, 
            dd_randomize_ops = ["DS","LMS_DS"],
            dd_order = "NATURAL",
            kernel_type_ops = [KernelDigShiftInvar])
    def _sample(self, n_min, n_max):
        xb = self.dd_obj.gen_samples(n_min=n_min,n_max=n_max,return_binary=True)
        xf = xb*2**(-self.kernel_obj.t)
        return xb,xf

    
