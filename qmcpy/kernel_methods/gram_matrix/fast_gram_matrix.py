from ._gram_matrix import _GramMatrix
from ...discrete_distribution import Lattice,DigitalNetB2,DiscreteDistribution
from ..kernel import KernelShiftInvar,KernelDigShiftInvar
from ..util import fft_bro_1d_radix2,ifft_bro_1d_radix2,fwht_1d_radix2
import numpy as np
import itertools
import warnings
    
class _FastGramMatrix(_GramMatrix):
    """
    >>> n = 2**3
    >>> d = 3
    >>> us = [np.array([int(b) for b in np.binary_repr(i,d)],dtype=bool) for i in range(2**d)]
    >>> lbetas = [
    ...     np.array([1,0,0]),
    ...     np.array([[0,1,0],[0,0,1]]),
    ...     [np.array([1,0,0]),np.array([0,1,0])],
    ...     [np.array([[1,0,0],[0,1,0],[0,0,1]]),np.array([[1,0,1],[0,1,0],[0,0,0]])]
    ...     ]
    >>> num_invertible = 0
    >>> lat_obj = Lattice(d,seed=7)
    >>> kernel_si = KernelShiftInvar(d,alpha=4)
    >>> dnb2_obj = DigitalNetB2(d,t_lms=32,alpha=2,seed=7)
    >>> kernel_dsi = KernelDigShiftInvar(d,alpha=4)
    >>> for iu1,iu2 in itertools.product(range(2**d),range(2**d)):
    ...     u1 = us[iu1]
    ...     u2 = us[iu2]
    ...     n1 = n if u1.sum()>0 else 1 
    ...     n2 = n if u2.sum()>0 else 1
    ...     for ib1,ib2 in itertools.product(range(len(lbetas)),range(len(lbetas))):
    ...         lbeta1s = lbetas[ib1]
    ...         lbeta2s = lbetas[ib2]
    ...         lc1s = 1.
    ...         lc2s = 1.
    ...         gm_lat = FastGramMatrixLattice(lat_obj,kernel_si,n1,n2,u1,u2,lbeta1s,lbeta2s,lc1s,lc2s)
    ...         gm_dnb2 = FastGramMatrixDigitalNetB2(dnb2_obj,kernel_dsi,n1,n2,u1,u2,lbeta1s,lbeta2s,lc1s,lc2s)
    ...         for gm in [gm_lat,gm_dnb2]:
    ...             gm._check()
    ...             gm_tall = gm.copy(max(n1//2,1),max(n2//4,1))
    ...             gm_tall._check()
    ...             gm_wide = gm.copy(max(n1//4,1),max(n2//2,1))
    ...             gm_wide._check()
    """
    def __init__(self, dd_obj, kernel_obj, n1, n2, u1, u2, lbeta1s, lbeta2s, lc1s, lc2s, noise, ft, ift, dd_type, dd_randomize_ops, dd_order, kernel_type_ops):
        super(_FastGramMatrix,self).__init__(kernel_obj,noise,lbeta1s,lbeta2s,lc1s,lc2s)
        assert any(isinstance(self.kernel_obj,kernel_type_op) for kernel_type_op in kernel_type_ops)
        self.ft = ft 
        self.ift = ift
        self.n1 = n1 
        self.n2 = n2 
        assert (self.n1&(self.n1-1))==0 and (self.n2&(self.n2-1))==0 and self.n1>0 and self.n2>0 # require n1 and n2 are powers of 2
        self.u1 = self.npt.ones(self.d,dtype=bool) if u1 is True else u1 
        self.u2 = self.npt.ones(self.d,dtype=bool) if u2 is True else u2
        assert self.u1.shape==(self.d,) and self.u2.shape==(self.d,) 
        assert (self.u1.sum()>0 or self.n1==1) and (self.u2.sum()>0 or self.n2==1)
        self.u1mu2 = self.u1*(~self.u2) 
        self.u2mu1 = self.u2*(~self.u1)
        self.u1au2 = self.u1*self.u2
        self.u1nu2 = (~self.u1)*(~self.u2)
        self.d_u1mu2 = self.u1mu2.sum()
        self.d_u2mu1 = self.u2mu1.sum()
        self.d_u1au2 = self.u1au2.sum()
        self.d_u1nu2 = self.u1nu2.sum()
        self.noise = noise
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
        inds = np.empty((self.t1,self.t2),dtype=object)
        idxs = np.empty((self.t1,self.t2),dtype=object)
        consts = np.empty((self.t1,self.t2),dtype=object)
        for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
            inds[tt1,tt2],idxs[tt1,tt2],consts[tt1,tt2] = self.kernel_obj.inds_idxs_consts(self.lbeta1s[tt1][:,None,:],self.lbeta2s[tt2][None,:,:])
        if self.d_u1nu2>0:
            delta_u2nu1 = self.kernel_obj.x1_ominus_x2(self.npt.zeros((1,1,self.d_u1nu2),dtype=self._x.dtype),self.npt.zeros((1,1,self.d_u1nu2),dtype=self._x.dtype)) # (1,1,self.d_u1nu2)
            self.scale_null = np.empty((self.t1,self.t2),dtype=object)
            for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                self.scale_null[tt1,tt2] = self.kernel_obj.eval_low_u_noscale(self.u1nu2,delta_u2nu1,inds[tt1,tt2],idxs[tt1,tt2],consts[tt1,tt2],self.m1[tt1],self.m2[tt2],1,1,self.d_u1nu2)[:,:,0,0] # (self.m1,self.m2)
        if self.d_u1mu2>0:
            delta_u1mu2 = self.kernel_obj.x1_ominus_x2(self._x[:self.n1,None,self.u1mu2],self.npt.zeros((1,1,self.d_u1mu2),dtype=self._x.dtype))
            self.k1l = np.empty((self.t1,self.t2),dtype=object)
            for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                self.k1l[tt1,tt2] = self.kernel_obj.eval_low_u_noscale(self.u1mu2,delta_u1mu2,inds[tt1,tt2],idxs[tt1,tt2],consts[tt1,tt2],self.m1[tt1],self.m2[tt2],self.n1,1,self.d_u1mu2)[:,:,:,0] # (self.m1,self.m2,self.n1)
        if self.d_u2mu1>0:
            delta_u2mu1 = self.kernel_obj.x1_ominus_x2(self.npt.zeros((1,1,self.d_u2mu1),dtype=self._x.dtype),self._x[None,:self.n2,self.u2mu1])
            self.k1r = np.empty((self.t1,self.t2),dtype=object)
            for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                self.k1r[tt1,tt2] = self.kernel_obj.eval_low_u_noscale(self.u2mu1,delta_u2mu1,inds[tt1,tt2],idxs[tt1,tt2],consts[tt1,tt2],self.m1[tt1],self.m2[tt2],1,self.n2,self.d_u2mu1)[:,:,0,:] # (self.m1,self.m2,self.n2)
        if self.d_u1au2>0:
            if self.n1==self.n2:
                self.vhs = "square"
                delta_u1au2 = self.kernel_obj.x1_ominus_x2(self._x[:n1,None,self.u1au2],self._x[None,[0],self.u1au2])
                k1 = np.empty((self.t1,self.t2),dtype=object)
                for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                    k1[tt1,tt2] = self.kernel_obj.eval_low_u_noscale(self.u1au2,delta_u1au2,inds[tt1,tt2],idxs[tt1,tt2],consts[tt1,tt2],self.m1[tt1],self.m2[tt2],self.n1,1,self.d_u1au2).transpose(0,1,3,2)
                # self.lam will be (self.m1,self.m2,1,self.n1)
            elif self.n1>self.n2:
                self.vhs = "tall"
                self.r = self.n1//self.n2
                delta_u1au2 = self.kernel_obj.x1_ominus_x2(self._x[:n1,None,self.u1au2],self._x[None,[0],self.u1au2])
                k1 = np.empty((self.t1,self.t2),dtype=object)
                for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                    k1[tt1,tt2] = self.kernel_obj.eval_low_u_noscale(self.u1au2,delta_u1au2,inds[tt1,tt2],idxs[tt1,tt2],consts[tt1,tt2],self.m1[tt1],self.m2[tt2],self.n1,1,self.d_u1au2).reshape((self.m1[tt1],self.m2[tt2],self.r,self.n2))
                # self.lam will be (self.m1,self.m2,self.r,self.n2)
            else: # self.n1<self.n2
                self.vhs = "wide"
                self.r = self.n2//self.n1
                delta_u1au2 = self.kernel_obj.x1_ominus_x2(self._x[:n1,None,self.u1au2],self._x[None,:self.n2:self.n1,self.u1au2])
                k1 = np.empty((self.t1,self.t2),dtype=object)
                for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                    k1[tt1,tt2] = self.kernel_obj.eval_low_u_noscale(self.u1au2,delta_u1au2,inds[tt1,tt2],idxs[tt1,tt2],consts[tt1,tt2],self.m1[tt1],self.m2[tt2],self.n1,self.r,self.d_u1au2).transpose(0,1,3,2)
                # self.lam will be (self.m1,self.m2,self.r,self.n1)
            if self.d_u1nu2>0:
                for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                    k1[tt1,tt2] = k1[tt1,tt2]*self.scale_null[tt1,tt2][:,:,None,None]
                # self.lam will have (m1,m2)=(1,1)
                delattr(self,"scale_null")
                self.d_u1nu2 = 0
            if self.d_u1mu2==0 and self.d_u2mu1==0:
                for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                    k1[tt1,tt2] = (self.lc1s[tt1][:,None,None,None]*self.lc2s[tt2][None,:,None,None]*k1[tt1,tt2]).sum((0,1))[None,None,:,:]
                self.lc1s = [self.npt.ones(1) for tt1 in range(self.t1)]
                self.lc2s = [self.npt.ones(1) for tt2 in range(self.t2)]
                self.m1 = np.ones(self.t1,dtype=int)
                self.m2 = np.ones(self.t2,dtype=int)
            self.lam = np.empty((self.t1,self.t2),dtype=object)
            for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                self.lam[tt1,tt2] = np.sqrt(self.n_min)*self.ft(k1[tt1,tt2])
        invertible_conds = [
            ( self.n1==self.n2, "require square matrices"),
            ( self.d_u1au2>0, "require a positive definite circulant factor"),
            ( self.t1==self.t2 and all((self.lbeta1s[tt1]==self.lbeta2s[tt1]).all() and (self.lc1s[tt1]==self.lc2s[tt1]).all() for tt1 in range(self.t1)), "require lbeta1s=lbeta2s and lc1s=lc2s"),
            ( (self.m1==1).all() and (self.m2==1).all(), "require there is only one derivative order (also satisfied when self.d_u1mu2==0 and self.d_u2mu1==0)"),
            ( (self.t1==1 and self.t2==1) or (self.d_u1mu2==0 and self.d_u2mu1==0), "Only allow more than one beta block when there are no left or right factors in each block"),
            ]
        super(_FastGramMatrix,self)._set_invertible_conds(invertible_conds)     
        self.size = (self.n1*self.t1,self.n2*self.t2)
    def _init_invertibile(self):
        lamblock = 1j*self.npt.empty((self.n1,self.t1,self.t1))
        for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
            lamblock[:,tt1,tt2] = self.lam[tt1,tt2][0,0,0]
        self.l_chol = self.cholesky(lamblock+self.noise*self.npt.eye(self.t1))
    def sample(self, n_min, n_max):
        assert hasattr(self,"dd_obj"), "no discrete distribution object available to sample from"
        _x,x = self._sample(n_min,n_max)
        return _x,x
    def get_full_gram_matrix(self):
        _xu1,_xu2 = self._x.copy(),self._x.copy()
        _xu1[:,~self.u1] = 0.
        _xu2[:,~self.u2] = 0.
        kfull = np.empty((self.t1,self.t2),dtype=object)
        for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
            kfull[tt1,tt2] = self.kernel_obj(_xu1[:self.n1,:],_xu2[:self.n2,:],self.lbeta1s[tt1],self.lbeta2s[tt2],self.lc1s_og[tt1],self.lc2s_og[tt2])
        return self.npt.vstack([self.npt.hstack(kfull[tt1]) for tt1 in range(self.t1)])
    def multiply(self, *args, **kwargs):
        return self.__matmul__(*args, **kwargs)
    def __matmul__(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None]
        assert y.ndim==2 and y.shape[0]==(self.n2*self.t2)
        v = y.shape[1] # y is (t2*n2,v)
        y = y*self.kernel_obj.scale
        y = y.T.reshape((v,self.t2,self.n2)) # (v,t2,n2)
        yfull = [y[:,tt2,:].reshape((v,1,1,self.n2)) for tt2 in range(self.t2)] # (v,1,1,n2)
        sfull = [0. for tt1 in range(self.t1)] # (v,n1)
        for tt1 in range(self.t1):
            for tt2 in range(self.t2): 
                y = yfull[tt2]
                if self.d_u1nu2>0:
                    y = y*self.scale_null[tt1,tt2][:,:,None] # (v,m1,m2,n2) since self.scale_null is (m1,m2)
                if self.d_u2mu1>0:
                    y = y*self.k1r[tt1,tt2] # (v,m1,m2,n2) since self.k1r is (m1,m2,n2)
                if self.d_u1au2>0:
                    if self.vhs=="square": # so n1=n2
                        yt = self.ft(y) # (v,m1,m2,n1) or (v,1,1,n1)
                        st = yt*self.lam[tt1,tt2][:,:,0,:] # (v,n2) since self.lam is (m1,m2,1,n1)
                        s = self.ift(st).real # (v,m1,m2,n1)
                    elif self.vhs=="tall": # so n1 = r*n2
                        yt = self.ft(y) # (v,m1,m2,n2) or (v,1,1,n2)
                        st = yt[:,:,:,None,:]*self.lam[tt1,tt2] # (v,m1,m2,r,n2) since self.lam is (m1,m2,r,n2)
                        s = self.ift(st).real.reshape((v,self.m1[tt1],self.m2[tt2],self.n1)) # (v,m1,m2,n1)
                    else: # self.vhs=="wide", so n2 = r*n1
                        yt = self.ft(y.reshape(v,y.shape[1],y.shape[2],self.r,self.n1)) # (v,m1,m2,r,n1) or (v,1,1,r,n2) since y is either (v,m1,m2,n2) or (v,1,1,n2)
                        st = (yt*self.lam[tt1,tt2]).sum(3) # (v,m1,m2,n1) since self.lam is (m1,m2,r,n1)
                        s = self.ift(st).real # (v,m1,m2,n1)
                else: # left multiply by matrix of ones
                    s = self.npt.tile(y.sum(-1)[:,:,:,None],(self.n1,)) # (v,m1,m2,n1)
                if self.d_u1mu2>0:
                    s = s*self.k1l[tt1,tt2] # (v,m1,m2,n1) since self.k1l is (m1,m2,n1)
                s = (self.lc1s[tt1][None,:,None,None]*self.lc2s[tt2][None,None,:,None]*s).sum((1,2)) # (v,n1)
                sfull[tt1] = sfull[tt1]+s
        sfull = self.npt.hstack(sfull) # (v,self.n1*self.t1) since each element of sfull is (v,self.n1)
        return sfull[0,:] if yogndim==1 else sfull.T
    def solve(self, y):
        assert self.invertible, self.invertible_error_msg
        if not hasattr(self,"l_chol"): 
            self._init_invertibile()
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None]
        assert y.ndim==2 and y.shape[0]==(self.n1*self.t1)
        v = y.shape[1] # y is (t1*n1,v)
        y = y/self.kernel_obj.scale
        y = y.T # (v,t1*n1)
        if self.t1==1:
            if self.d_u1mu2>0:
                y = y/self.k1l[0,0][0,0] # (v,self.n1) since self.k1l is (1,1,n1)
            if self.d_u1au2>0:
                yt = self.ft(y) # (v,self.n1)
                st = yt/self.lam[0,0][0,0,0] # (v,self.n1) since self.lam is (1,1,self.n1)
                s = self.ift(st).real # (v,self.n1)
            if self.d_u2mu1>0:
                s = s/self.k1r[0,0][0,0] # (v,self.n1) since self.k1r is (1,1,n1)
        else:
            y = y.reshape((v,self.t1,self.n1)) # (v,t1,n1)
            yt = self.ft(y) # (v,t1,n1)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",".*Casting complex values to real discards the imaginary part*")
                for i in range(self.n1): # probably a better vectorized way to do this
                    # solve systems based on Cholesky decompositions, with self.l_chol is (n1,t1,t1)
                    yt[:,:,i] = self.cho_solve(self.l_chol[i],yt[:,:,i].T).T
            s = self.ift(yt).real # (v,t1,n1)
            s = s.reshape((v,self.t1*self.n1))
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
    def _mult_check(self, y, gmatfull):
        assert np.allclose(self@y[:,0],gmatfull@y[:,0],atol=1e-12)
        assert np.allclose(self@y,gmatfull@y,atol=1e-12)
    def _solve_check(self, y, gmatfull):
        if not self.invertible: return
        assert np.allclose(self.solve(y[:,0]),np.linalg.solve(gmatfull,y[:,0]),rtol=2.5e-2)
        assert np.allclose(self.solve(y),np.linalg.solve(gmatfull,y),rtol=2.5e-2)
    def _check(self):
        rng = np.random.Generator(np.random.SFC64(7))
        y = rng.uniform(size=(self.n2*self.t2,2))
        gmatfull = self.get_full_gram_matrix()
        self._mult_check(y,gmatfull)
        self._solve_check(y,gmatfull)

class FastGramMatrixLattice(_FastGramMatrix):
    """
    Fast Gram matrix operations using lattice points and shift invariant kernels 
    """
    def __init__(self, dd_obj, kernel_obj, n1, n2, u1=True, u2=True, lbeta1s=0, lbeta2s=0, lc1s=1., lc2s=1., noise=1e-8):
        """
        Args:
            dd_obj (Lattice): requires randomize='SHIFT' and order="NATURAL"
            kernel_obj (KernelShiftInvar): shift invariant kernel
            n1 (int): first number of points
            n2 (int): second number of points
            u1 (np.ndarray or torch.Tensor): length d bool vector of first active dimensions 
            u2 (np.ndarray or torch.Tensor): length d bool vector of second active dimensions
            lbeta1s (list of either np.ndarray or torch.Tensor): list of (m1[l],d) arrays of first derivative orders
            lbeta1s (list of either np.ndarray or torch.Tensor): list of (m2[l],d) arrays of first derivative orders
            c1s (list of either np.ndarray or torch.Tensor): list of length m1[l] vectors of derivative coefficients 
            c2s (list of either np.ndarray or torch.Tensor): list of length m2[l] vector of derivative coefficients
            noise (float): nugget term 
        """
        super(FastGramMatrixLattice,self).__init__(dd_obj,kernel_obj,n1,n2,u1,u2,lbeta1s,lbeta2s,lc1s,lc2s,noise,
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
    def __init__(self, dd_obj, kernel_obj, n1, n2, u1=True, u2=True, lbeta1s=0, lbeta2s=0, lc1s=1., lc2s=1., noise=1e-8):
        """
        Args:
            dd_obj (DigitalNetB2): requires randomize='LMS_DS' and graycode=False
            kernel_obj (KernelDigShiftInvar): digitally shift invariant kernel
            n1 (int): first number of points
            n2 (int): second number of points
            u1 (np.ndarray or torch.Tensor): length d bool vector of first active dimensions 
            u2 (np.ndarray or torch.Tensor): length d bool vector of second active dimensions
            lbeta1s (list of either np.ndarray or torch.Tensor): list of (m1[l],d) arrays of first derivative orders
            lbeta1s (list of either np.ndarray or torch.Tensor): list of (m2[l],d) arrays of first derivative orders
            c1s (list of either np.ndarray or torch.Tensor): list of length m1[l] vectors of derivative coefficients 
            c2s (list of either np.ndarray or torch.Tensor): list of length m2[l] vector of derivative coefficients
            noise (float): nugget term 
        """
        if isinstance(dd_obj,DigitalNetB2):
            kernel_obj.set_t(dd_obj.t_lms)
        super(FastGramMatrixDigitalNetB2,self).__init__(dd_obj,kernel_obj,n1,n2,u1,u2,lbeta1s,lbeta2s,lc1s,lc2s,noise,
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
