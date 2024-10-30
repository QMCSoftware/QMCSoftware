from ..discrete_distribution import Lattice,DigitalNetB2,DiscreteDistribution
from .kernels import KernelShiftInvar,KernelDigShiftInvar
from .fast_transforms import fft_bro_1d_radix2,ifft_bro_1d_radix2,fwht_1d_radix2
import numpy as np

def _mult_check(gm):
    y = np.vstack([np.sin(2*np.pi*gm.x[:gm.n2]).prod(1),np.cos(2*np.pi*gm.x[:gm.n2]).prod(1)]).T
    gmatfull = gm.get_full_gram_matrix()
    assert np.allclose(gm@y[:,0],gmatfull@y[:,0],atol=1e-12)
    assert np.allclose(gm@y,gmatfull@y,atol=1e-12)

def _solve_check(gm):
    y = np.vstack([np.sin(2*np.pi*gm.x[:gm.n2]).prod(1),np.cos(2*np.pi*gm.x[:gm.n2]).prod(1)]).T
    gmatfull = gm.get_full_gram_matrix()
    assert np.allclose(gm.solve(y[:,0]),np.linalg.solve(gmatfull,y[:,0]),atol=1e-12)
    assert np.allclose(gm.solve(y),np.linalg.solve(gmatfull,y),atol=1e-12)
    
class _FastGramMatrix(object):
    """
    >>> n = 2**5
    >>> d = 3
    >>> kernel_si = KernelShiftInvar(d)
    >>> kernel_dsi = KernelDigShiftInvar(d)
    >>> gm_lat_og = FastGramMatrixLattice(Lattice(d,seed=7),kernel_si,n,n)
    >>> gm_lat_sq = FastGramMatrixLattice((gm_lat_og._x,gm_lat_og.x),kernel_si,n//2,n//2)
    >>> gm_dnb2_og = FastGramMatrixDigitalNetB2(DigitalNetB2(d,seed=7),kernel_dsi,n,n)
    >>> gm_dnb2_sq = FastGramMatrixDigitalNetB2((gm_dnb2_og._x,gm_dnb2_og.x),kernel_dsi,n//2,n//2)
    >>> for gm in [gm_lat_og,gm_lat_sq,gm_dnb2_og,gm_dnb2_sq]:
    ...     _mult_check(gm)
    ...     _solve_check(gm)
    >>> gm_lat_tall = FastGramMatrixLattice((gm_lat_og._x,gm_lat_og.x),kernel_si,n//2,n//4)
    >>> gm_lat_wide = FastGramMatrixLattice((gm_lat_og._x,gm_lat_og.x),kernel_si,n//4,n//2)
    >>> gm_dnb2_tall = FastGramMatrixDigitalNetB2((gm_dnb2_og._x,gm_dnb2_og.x),kernel_dsi,n//2,n//4)
    >>> gm_dnb2_wide = FastGramMatrixDigitalNetB2((gm_dnb2_og._x,gm_dnb2_og.x),kernel_dsi,n//4,n//2)
    >>> for gm in [gm_lat_tall,gm_lat_wide,gm_dnb2_tall,gm_dnb2_wide]:
    ...     _mult_check(gm)
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
            self.lam = np.sqrt(self.n1)*self.ft(k1) # is (self.n1,)
        elif self.n1>self.n2:
            self.vhs = "tall"
            self.r = self.n1//self.n2
            k1 = self.kernel_obj(self._x[:n1,:],self._x[[0],:])[:,0].reshape((self.r,self.n2))
            self.lam = np.sqrt(self.n2)*self.ft(k1) # is (self.r,self.n2)
        else: # self.n1<self.n2
            self.vhs = "wide"
            self.r = self.n2//self.n1
            k1 = self.kernel_obj(self._x[:n1,:],self._x[:self.n2:self.n1,:]).T
            self.lam = np.sqrt(self.n1)*self.ft(k1) # is (self.r,n1)
    def sample(self, n_min, n_max):
        assert hasattr(self,"dd_obj"), "no discrete distribution object available to sample from"
        _x,x = self._sample(n_min,n_max)
        return _x,x
    def get_full_gram_matrix(self):
        return self.kernel_obj(self._x[:self.n1,:],self._x[:self.n2,:])
    def multiply(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None]
        assert y.ndim==2 and y.shape[0]==self.n2
        m = y.shape[1] # y is (self.n2,m)
        if self.vhs=="square": # so self.n1=self.n2
            yt = self.ft(y.T) # is (m,self.n2)
            st = yt*self.lam # is (m,self.n2) since self.lam is (self.n2,)
            s = self.ift(st).real.T # s is (self.n2,m)
        elif self.vhs=="tall": # so self.n1 = self.r*self.n2
            yt = self.ft(y.T) # is (m,self.n2)
            st = yt[:,None,:]*self.lam # is (m,self.r,self.n2) since self.lam is (self.r,self.n2)
            s = self.ift(st).real.reshape((m,self.n1)).T # is (self.n1,m)
        else: #self.vhs=="wide", so self.n2 = self.r*self.n1
            yt = self.ft(y.T.reshape(m,self.r,self.n1)) # is (m,self.r,self.n1)
            st = (yt*self.lam).sum(1) # is (m,self.n1) since self.lam is (self.r,self.n1)
            s = self.ift(st).real.T # is (self.n1,m)
        return s[:,0] if yogndim==1 else s
    def __matmul__(self, y):
        return self.multiply(y)
    def solve(self, y):
        assert self.vhs=="square", "cannot solve system in non-square matrix"
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None]
        assert y.ndim==2 and y.shape[0]==self.n1
        s = self.ift(self.ft(y.T)/self.lam).real.T
        return s[:,0] if yogndim==1 else s

class FastGramMatrixLattice(_FastGramMatrix):
    """
    Fast Gram matrix operations using lattice points and shift invariant kernels 
    """
    def __init__(self, lat_obj_or__x_x, kernel_si, n1, n2, noise=0.):
        """
        Args:
            lat_obj_or__x_x (Lattice or tuple): lattice discrete distribution with randomize='shift' 
                or a tuple (fgml._x,fgml.x) where isinstance(fgml,FastGramMatrixLattice)
            kernel_si (KernelShiftInvar): shift invariant kernel
            n1 (int): first number of points
            n2 (int): second number of points
            noise (float): nugget term 
        """
        super(FastGramMatrixLattice,self).__init__(lat_obj_or__x_x,kernel_si,n1,n2,noise,
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
    def __init__(self, dnb2_obj_or__x_x, kernel_dsi, n1, n2, noise=0.):
        """
        Args:
            dnb2_obj_or__x_x (DigitalNetB2 or tuple): lattice discrete distribution with randomize='shift' 
                or a tuple (fgml._x,fgml.x) where isinstance(fgml,FastGramMatrixDigitalNetB2)
            kernel_si (KernelShiftInvar): shift invariant kernel
            n1 (int): first number of points
            n2 (int): second number of points
            noise (float): nugget term 
        """
        if isinstance(dnb2_obj_or__x_x,DigitalNetB2):
            kernel_dsi.set_t(dnb2_obj_or__x_x.t_lms)
        super(FastGramMatrixDigitalNetB2,self).__init__(dnb2_obj_or__x_x,kernel_dsi,n1,n2,noise,
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

    
