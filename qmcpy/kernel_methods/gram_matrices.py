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
    assert np.allclose(gm.solve(y[:,0]),np.linalg.solve(gmatfull,y[:,0]),rtol=2.5e-2)
    assert np.allclose(gm.solve(y),np.linalg.solve(gmatfull,y),rtol=2.5e-2)
    
class _FastGramMatrix(object):
    """
    >>> n = 2**5
    >>> d = 3
    >>> u1 = np.array([False,True,True])
    >>> u2 = np.array([False,True,False])
    >>> beta1 = np.array([0,1,0],dtype=int)
    >>> beta2 = np.array([0,1,0],dtype=int)
    >>> kernel_si = KernelShiftInvar(d,alpha=4)
    >>> kernel_dsi = KernelDigShiftInvar(d,alpha=4)
    >>> gm_lat_og = FastGramMatrixLattice(Lattice(d,seed=7),kernel_si,n,n,u1,u2,beta1,beta2)
    >>> gm_lat_sq = gm_lat_og.copy(n//2,n//2)
    >>> gm_dnb2_og = FastGramMatrixDigitalNetB2(DigitalNetB2(d,seed=7),kernel_dsi,n,n,u1,u2,beta1,beta2)
    >>> gm_dnb2_sq = gm_dnb2_og.copy(n//2,n//2)
    >>> for gm in [gm_lat_og,gm_lat_sq,gm_dnb2_og,gm_dnb2_sq]:
    ...     _mult_check(gm)
    ...     if gm.invertible:
    ...         _solve_check(gm)
    >>> gm_lat_tall = gm_lat_og.copy(n//2,n//4)
    >>> gm_lat_wide = gm_lat_og.copy(n//4,n//2)
    >>> gm_dnb2_tall = gm_dnb2_og.copy(n//2,n//4)
    >>> gm_dnb2_wide = gm_dnb2_og.copy(n//4,n//2)
    >>> for gm in [gm_lat_tall,gm_lat_wide,gm_dnb2_tall,gm_dnb2_wide]:
    ...     _mult_check(gm)
    """
    def __init__(self, dd_obj, kernel_obj, n1, n2, u1, u2, beta1s, beta2s, c1s, c2s, noise, ft, ift, dd_type, dd_randomize_ops, dd_order, kernel_type_ops):
        self.kernel_obj = kernel_obj
        self.d = self.kernel_obj.d
        self.npt = self.kernel_obj.npt
        self.ft = ft 
        self.ift = ift
        self.n1 = n1 
        self.n2 = n2 
        assert (self.n1&(self.n1-1))==0 and (self.n2&(self.n2-1))==0 and self.n1>0 and self.n2>0 # require n1 and n2 are powers of 2
        self.u1 = self.npt.ones(self.d,dtype=bool) if u1 is True else u1 
        self.u2 = self.npt.ones(self.d,dtype=bool) if u2 is True else u2
        assert self.u1.shape==(self.d,) and self.u2.shape==(self.d,) and self.u1.sum()>0 and self.u2.sum()>0
        self.u1mu2 = self.u1*(~self.u2) 
        self.u2mu1 = self.u2*(~self.u1)
        self.u1au2 = self.u1*self.u2
        self.u1nu2 = (~self.u1)*(~self.u2)
        self.d_u1mu2 = self.u1mu2.sum()
        self.d_u2mu1 = self.u2mu1.sum()
        self.d_u1au2 = self.u1au2.sum()
        self.d_u1nu2 = self.u1nu2.sum()
        self.noise = noise
        assert self.noise==0
        #assert isinstance(self.noise,float) and self.noise>=0
        if isinstance(dd_obj,DiscreteDistribution):
            self.dd_obj = dd_obj
            assert isinstance(self.dd_obj,dd_type) and self.dd_obj.d==self.d 
            assert self.dd_obj.replications==1
            assert self.dd_obj.randomize in dd_randomize_ops and self.dd_obj.order==dd_order
            self.n_max = max(self.n1,self.n2)
            self._x,self.x = self.sample(0,self.n_max)
        else:
            assert isinstance(dd_obj,tuple) and len(dd_obj)==2
            self._x,self.x = dd_obj
            self.n_max = len(self._x)
            assert self.n_max>=self.n1 and self.n_max>=self.n2
            assert self._x.shape==(self.n_max,self.kernel_obj.d) and self.x.shape==(self.n_max,self.kernel_obj.d)
        self.n_min = min(self.n1,self.n2)
        assert any(isinstance(kernel_obj,kernel_type_op) for kernel_type_op in kernel_type_ops)
        self.beta1s,self.beta2s,self.c1s,self.c2s,self.m1,self.m2 = self.kernel_obj._parse_betas_cs(beta1s,beta2s,c1s,c2s)
        inds,idxs,consts = self.kernel_obj.inds_idxs_consts(self.beta1s[:,None,:],self.beta2s[None,:,:])
        if self.d_u1nu2>0:
            delta_u2nu1 = self.kernel_obj.x1_ominus_x2(self.npt.zeros((1,1,self.d_u1nu2),dtype=self._x.dtype),self.npt.zeros((1,1,self.d_u1nu2),dtype=self._x.dtype)) # (1,1,self.d_u1nu2)
            self.scale_null = self.kernel_obj.eval_low_u_noscale(self.u1nu2,delta_u2nu1,inds,idxs,consts,self.m1,self.m2,1,1,self.d_u1nu2)[:,:,0,0] # (self.m1,self.m2)
        if self.d_u1mu2>0:
            delta_u1mu2 = self.kernel_obj.x1_ominus_x2(self._x[:self.n1,None,self.u1mu2],self.npt.zeros((1,1,self.d_u1mu2),dtype=self._x.dtype))
            self.k1l = self.kernel_obj.eval_low_u_noscale(self.u1mu2,delta_u1mu2,inds,idxs,consts,self.m1,self.m2,self.n1,1,self.d_u1mu2)[:,:,:,0] # (self.m1,self.m2,self.n1)
        if self.d_u2mu1>0:
            delta_u2mu1 = self.kernel_obj.x1_ominus_x2(self.npt.zeros((1,1,self.d_u2mu1),dtype=self._x.dtype),self._x[None,:self.n2,self.u2mu1])
            self.k1r = self.kernel_obj.eval_low_u_noscale(self.u2mu1,delta_u2mu1,inds,idxs,consts,self.m1,self.m2,1,self.n2,self.d_u2mu1)[:,:,0,:] # (self.m1,self.m2,self.n2)
        if self.d_u1au2>0:
            if self.n1==self.n2:
                self.vhs = "square"
                delta_u1au2 = self.kernel_obj.x1_ominus_x2(self._x[:n1,None,self.u1au2],self._x[None,[0],self.u1au2])
                k1 = self.kernel_obj.eval_low_u_noscale(self.u1au2,delta_u1au2,inds,idxs,consts,self.m1,self.m2,self.n1,1,self.d_u1au2)[:,:,:,0]
                # self.lam will be (self.m1,self.m2,self.n1)
            elif self.n1>self.n2:
                self.vhs = "tall"
                self.r = self.n1//self.n2
                delta_u1au2 = self.kernel_obj.x1_ominus_x2(self._x[:n1,None,self.u1au2],self._x[None,[0],self.u1au2])
                k1 = self.kernel_obj.eval_low_u_noscale(self.u1au2,delta_u1au2,inds,idxs,consts,self.m1,self.m2,self.n1,1,self.d_u1au2).reshape((self.m1,self.m2,self.r,self.n2))
                # self.lam will be (self.m1,self.m2,self.r,self.n2)
            else: # self.n1<self.n2
                self.vhs = "wide"
                self.r = self.n2//self.n1
                delta_u1au2 = self.kernel_obj.x1_ominus_x2(self._x[:n1,None,self.u1au2],self._x[None,:self.n2:self.n1,self.u1au2])
                k1 = self.kernel_obj.eval_low_u_noscale(self.u1au2,delta_u1au2,inds,idxs,consts,self.m1,self.m2,self.n1,self.r,self.d_u1au2).transpose(0,1,3,2)
                # self.lam will be (self.m1,self.m2,self.r,self.n1)
            if self.d_u1nu2>0:
                k1 = k1*self.scale_null
                delattr(self,"scale_null")
                self.d_u1nu2 = 0
            if self.d_u1mu2==0 and self.d_u2mu1==0:
                k1 = (self.c1s[:,None,None,None]*self.c2s[None,:,None,None]*k1).sum((0,1))[None,None,:,:]
                self.c1s = self.npt.ones(1)
                self.c2s = self.npt.ones(1)
                self.m1 = 1 
                self.m2 = 1
            self.lam = np.sqrt(self.n_min)*self.ft(k1) 
        self.invertible = self.d_u1au2>0 and self.vhs=="square" and (self.beta1s==self.beta2s).all() and self.m1==1 and self.m2==1
    def sample(self, n_min, n_max):
        assert hasattr(self,"dd_obj"), "no discrete distribution object available to sample from"
        _x,x = self._sample(n_min,n_max)
        return _x,x
    def get_full_gram_matrix(self):
        _xu1,_xu2 = self._x.copy(),self._x.copy()
        _xu1[:,~self.u1] = 0.
        _xu2[:,~self.u2] = 0.
        return self.kernel_obj(_xu1[:self.n1,:],_xu2[:self.n2,:],self.beta1s,self.beta2s,self.c1s,self.c2s)
    def multiply(self, *args, **kwargs):
        return self.__matmul__(*args, **kwargs)
    def __matmul__(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None]
        assert y.ndim==2 and y.shape[0]==self.n2
        v = y.shape[1] # y is (n2,v)
        y = (y*self.kernel_obj.scale).T[:,None,None,:] # (v,1,1,n2)
        if self.d_u1nu2>0:
            y = y*self.scale_null[:,:,None] # (v,m1,m2,n2) since self.scale_null is (m1,m2)
        if self.d_u2mu1>0:
            y = y*self.k1r # (v,m1,m2,n2) since self.k1r is (m1,m2,n2)
        if self.d_u1au2>0:
            if self.vhs=="square": # so n1=n2
                yt = self.ft(y) # (v,m1,m2,n1) or (v,1,1,n1)
                st = yt*self.lam # (v,n2) since self.lam is (m1,m2,n1)
                s = self.ift(st).real # (v,m1,m2,n1)
            elif self.vhs=="tall": # so n1 = r*n2
                yt = self.ft(y) # (v,m1,m2,n2) or (v,1,1,n2)
                st = yt[:,:,:,None,:]*self.lam # (v,m1,m2,r,n2) since self.lam is (m1,m2,r,n2)
                s = self.ift(st).real.reshape((v,self.m1,self.m2,self.n1)) # (v,m1,m2,n1)
            else: # self.vhs=="wide", so n2 = r*n1
                yt = self.ft(y.reshape(v,y.shape[1],y.shape[2],self.r,self.n1)) # (v,m1,m2,r,n1) or (v,1,1,r,n2) since y is either (v,m1,m2,n2) or (v,1,1,n2)
                st = (yt*self.lam).sum(3) # (v,m1,m2,n1) since self.lam is (m1,m2,r,n1)
                s = self.ift(st).real # (v,m1,m2,n1)
        else: # left multiply by matrix of ones
            s = self.npt.tile(y.sum(-1)[:,:,:,None],(self.n1,)) # (v,m1,m2,n1)
        if self.d_u1mu2>0:
            s = s*self.k1l # (v,m1,m2,n1) since self.k1l is (m1,m2,n1)
        s = (self.c1s[None,:,None,None]*self.c2s[None,None,:,None]*s).sum((1,2)) # (v,n1)
        return s[0,:] if yogndim==1 else s.T
    def solve(self, y):
        assert self.invertible, """
            Require square matrix i.e. n1=n2. 
            Require u1,u2 share an active column i.e. (u1*u2).sum()>0. 
            Require beta1s==beta2s.
            Require m1=m2 i.e. either beta1s.shape[0]==beta2s.shape[0]==1 or ( (u1==True).all() and (u2==True).all() )
        """
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None]
        assert y.ndim==2 and y.shape[0]==self.n1
        y = (y/self.kernel_obj.scale).T # (v,self.n1)
        if self.d_u1nu2>0:
            y = y/self.scale_null[0,0] # (v,self.n1) since self.scale_null is (1,1)
        if self.d_u1mu2>0:
            y = y/self.k1l[0,0] # (v,self.n1) since self.k1l is (1,1,n1)
        if self.d_u1au2>0:
            yt = self.ft(y) # (v,self.n1)
            st = yt/self.lam[0,0] # (v,self.n1) since self.lam is (1,1,self.n1)
            s = self.ift(st).real # (v,self.n1)
        if self.d_u2mu1>0:
            s = s/self.k1r[0,0] # (v,self.n1) since self.k1r is (1,1,n1)
        return s[0,:] if yogndim==1 else s.T
    def copy(self, n1=None, n2=None):
        if n1 is None: n1 = self.n1 
        if n2 is None: n2 = self.n2
        return type(self)(
            dd_obj = (self._x,self.x),
            kernel_obj = self.kernel_obj,
            n1 = self.n1 if n1 is None else n1,
            n2 = self.n2 if n2 is None else n2,
            u1 = self.u1,
            u2 = self.u2,
            noise = self.noise)

class FastGramMatrixLattice(_FastGramMatrix):
    """
    Fast Gram matrix operations using lattice points and shift invariant kernels 
    """
    def __init__(self, dd_obj, kernel_obj, n1, n2, u1=True, u2=True, beta1=0, beta2=0, c1s=1., c2s=1., noise=0.):
        """
        Args:
            dd_obj (Lattice): requires randomize='SHIFT' and order="NATURAL"
            kernel_obj (KernelShiftInvar): shift invariant kernel
            n1 (int): first number of points
            n2 (int): second number of points
            u1 (np.ndarray or torch.Tensor): length d bool vector of first active dimensions 
            u2 (np.ndarray or torch.Tensor): length d bool vector of second active dimensions
            beta1 (np.ndarray or torch.Tensor): length d vector of first derivative orders
            beta2 (np.ndarray or torch.Tensor): length d vector of second derivative orders
            c1s (np.ndarray or torch.Tensor): length m1 vector of derivative coefficients 
            c2s (np.ndarray or torch.Tensor): length m2 vector of derivative coefficients
            noise (float): nugget term 
        """
        super(FastGramMatrixLattice,self).__init__(dd_obj,kernel_obj,n1,n2,u1,u2,beta1,beta2,c1s,c2s,noise,
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
    def __init__(self, dd_obj, kernel_obj, n1, n2, u1=True, u2=True, beta1=0, beta2=0, c1s=1., c2s=1., noise=0.):
        """
        Args:
            dd_obj (DigitalNetB2): requires randomize='LMS_DS' and graycode=False
            kernel_obj (KernelDigShiftInvar): digitally shift invariant kernel
            n1 (int): first number of points
            n2 (int): second number of points
            u1 (np.ndarray or torch.Tensor): length d bool vector of first active dimensions 
            u2 (np.ndarray or torch.Tensor): length d bool vector of second active dimensions
            beta1 (np.ndarray or torch.Tensor): length d vector of first derivative orders
            beta2 (np.ndarray or torch.Tensor): length d vector of second derivative orders
            c1s (np.ndarray or torch.Tensor): length m1 vector of derivative coefficients 
            c2s (np.ndarray or torch.Tensor): length m2 vector of derivative coefficients
            noise (float): nugget term 
        """
        if isinstance(dd_obj,DigitalNetB2):
            kernel_obj.set_t(dd_obj.t_lms)
        super(FastGramMatrixDigitalNetB2,self).__init__(dd_obj,kernel_obj,n1,n2,u1,u2,beta1,beta2,c1s,c2s,noise,
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

    
