from ._gram_matrix import _GramMatrix
from ...discrete_distribution import Lattice,DigitalNetB2,DiscreteDistribution
from ..kernel import KernelShiftInvar,KernelDigShiftInvar
from ..fast_transforms import fftbr,ifftbr,fwht
try:
    from ..fast_transforms import fftbr_torch,ifftbr_torch,fwht_torch
except:
    pass
import numpy as np
import itertools
import warnings
    
class _FastGramMatrix(_GramMatrix):
    def __init__(self, kernel_obj, dd_obj, n1, n2, u1, u2, lbeta1s, lbeta2s, lc1s, lc2s, noise, adaptive_noise, _pregenerated_x__x):
        super(_FastGramMatrix,self).__init__(kernel_obj,noise,lbeta1s,lbeta2s,lc1s,lc2s)
        self.n1 = n1 
        self.n2 = n2 
        assert (self.n1&(self.n1-1))==0 and (self.n2&(self.n2-1))==0 and self.n1>0 and self.n2>0 # require n1 and n2 are powers of 2
        self.u1 = self.npt.ones(self.d,dtype=bool,**self.ckwargs) if u1 is True else u1 
        self.u2 = self.npt.ones(self.d,dtype=bool,**self.ckwargs) if u2 is True else u2
        assert self.u1.shape==(self.d,) and self.u2.shape==(self.d,) 
        assert (self.u1.sum()>0 or self.n1==1) and (self.u2.sum()>0 or self.n2==1)
        self.u1mu2 = self.u1*(~self.u2) 
        self.u2mu1 = self.u2*(~self.u1)
        self.u1au2 = self.u1*self.u2
        self.u1nu2 = (~self.u1)*(~self.u2)
        self.d_u1mu2 = self.u1mu2.sum()
        self.d_u2mu1 = self.u2mu1.sum()
        self.d_u1au2 = self.u1au2.sum()
        self.d_u1nu2_og = self.d_u1nu2 = self.u1nu2.sum()
        if self.d_u1nu2==self.d: assert self.n1==self.n2==1
        self.dd_obj = dd_obj
        if _pregenerated_x__x is None:
            self.n_max = max(self.n1,self.n2)
            self._x,self.x = self.sample(0,self.n_max)
        else:
            self._x,self.x = _pregenerated_x__x
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
            delta_u2nu1 = self.kernel_obj.x1_ominus_x2(self.npt.zeros((1,1,self.d_u1nu2),dtype=self._x.dtype,**self.ckwargs),self.npt.zeros((1,1,self.d_u1nu2),dtype=self._x.dtype,**self.ckwargs)) # (1,1,self.d_u1nu2)
            self.scale_null = np.empty((self.t1,self.t2),dtype=object)
            for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                self.scale_null[tt1,tt2] = self.kernel_obj.eval_low_u_noscale(self.u1nu2,delta_u2nu1,inds[tt1,tt2],idxs[tt1,tt2],consts[tt1,tt2],self.m1[tt1],self.m2[tt2],1,1,self.d_u1nu2)[:,:,0,0] # (self.m1,self.m2)
        if self.d_u1mu2>0:
            delta_u1mu2 = self.kernel_obj.x1_ominus_x2(self._x[:self.n1,None,self.u1mu2],self.npt.zeros((1,1,self.d_u1mu2),dtype=self._x.dtype,**self.ckwargs))
            self.k1l = np.empty((self.t1,self.t2),dtype=object)
            for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                self.k1l[tt1,tt2] = self.kernel_obj.eval_low_u_noscale(self.u1mu2,delta_u1mu2,inds[tt1,tt2],idxs[tt1,tt2],consts[tt1,tt2],self.m1[tt1],self.m2[tt2],self.n1,1,self.d_u1mu2)[:,:,:,0] # (self.m1,self.m2,self.n1)
        if self.d_u2mu1>0:
            delta_u2mu1 = self.kernel_obj.x1_ominus_x2(self.npt.zeros((1,1,self.d_u2mu1),dtype=self._x.dtype,**self.ckwargs),self._x[None,:self.n2,self.u2mu1])
            self.k1r = np.empty((self.t1,self.t2),dtype=object)
            for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                self.k1r[tt1,tt2] = self.kernel_obj.eval_low_u_noscale(self.u2mu1,delta_u2mu1,inds[tt1,tt2],idxs[tt1,tt2],consts[tt1,tt2],self.m1[tt1],self.m2[tt2],1,self.n2,self.d_u2mu1)[:,:,0,:] # (self.m1,self.m2,self.n2)
        if self.d_u1au2>0 or self.d_u1nu2==self.d:
            if self.n1==self.n2:
                self.vhs = "square"
                delta_u1au2 = self.kernel_obj.x1_ominus_x2(self._x[:n1,None,self.u1au2],self._x[None,[0],self.u1au2])
                k1 = np.empty((self.t1,self.t2),dtype=object)
                for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                    k1[tt1,tt2] = self.transpose_func(self.kernel_obj.eval_low_u_noscale(self.u1au2,delta_u1au2,inds[tt1,tt2],idxs[tt1,tt2],consts[tt1,tt2],self.m1[tt1],self.m2[tt2],self.n1,1,self.d_u1au2),(0,1,3,2))
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
                    k1[tt1,tt2] = self.transpose_func(self.kernel_obj.eval_low_u_noscale(self.u1au2,delta_u1au2,inds[tt1,tt2],idxs[tt1,tt2],consts[tt1,tt2],self.m1[tt1],self.m2[tt2],self.n1,self.r,self.d_u1au2),(0,1,3,2))
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
                self.lc1s = [self.npt.ones(1,dtype=float,**self.ckwargs) for tt1 in range(self.t1)]
                self.lc2s = [self.npt.ones(1,dtype=float,**self.ckwargs) for tt2 in range(self.t2)]
                self.m1 = np.ones(self.t1,dtype=int)
                self.m2 = np.ones(self.t2,dtype=int)
            self.lam = np.empty((self.t1,self.t2),dtype=object)
            for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
                self.lam[tt1,tt2] = np.sqrt(self.n_min)*self.ft(k1[tt1,tt2])
        self.lc1slc2s = np.empty((self.t1,self.t2),dtype=object) 
        for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
            self.lc1slc2s[tt1,tt2] = self.lc1s[tt1][None,:,None,None]*self.lc2s[tt2][None,None,:,None]
        if hasattr(self,"lam"):
            self.lam *= self.kernel_obj.scale 
        elif hasattr(self,"k1l"):
            self.k1l *= self.kernel_obj.scale 
        else:
            assert hasattr(self,"k1r")
            self.k1r *= self.kernel_obj.scale 
        self.size = (self.n1*self.t1,self.n2*self.t2)
        invertible_conds = [
            ( self.n1==self.n2, "require square matrices"),
            ( self.d_u1au2>0 or self.n1==1, "require a positive definite circulant factor"),
            ( self.t1==self.t2 and all((self.lbeta1s[tt1]==self.lbeta2s[tt1]).all() and (self.lc1s[tt1]==self.lc2s[tt1]).all() for tt1 in range(self.t1)), "require lbeta1s=lbeta2s and lc1s=lc2s"),
            ( (self.m1==1).all() and (self.m2==1).all(), "require there is only one derivative order (also satisfied when self.d_u1mu2==0 and self.d_u2mu1==0)"),
            ( (self.t1==1 and self.t2==1 and self.noise==0) or (self.d_u1mu2==0 and self.d_u2mu1==0), "Only allow more than one beta block when there are no left or right factors in each block"),
            ]
        super(_FastGramMatrix,self)._set_invertible_conds(invertible_conds)
        self.adaptive_noise = adaptive_noise
        if self.invertible:
            self.k00diags = self.npt.ones(self.t1) 
            for tt1 in range(self.t1):
                self.k00diags[tt1] = k1[tt1,tt1][0,0,0,0]
            if self.adaptive_noise:
                assert (self.lbeta1s[0]==0.).all() and (self.lbeta1s[0].shape==(1,self.d)) and (self.lc1s_og[0]==1.).all() and (self.lc1s_og[0].shape==(1,))
                trace0 = self.k00diags[0] 
                for tt1 in range(self.t1):
                    trace = self.k00diags[tt1] 
                    trace_ratio = trace/trace0 
                    self.lam[tt1,tt1][0,0,0,:] += self.noise*trace_ratio
            else:
                for tt1 in range(self.t1):
                    self.lam[tt1,tt1][0,0,0,:] += self.noise # lam is (m1,m2,1,n1) = (1,1,1,n1)
    def _set__x1__x2(self):
        self._x1,self._x2 = self.clone(self._x),self.clone(self._x)
        self._x1[:,~self.u1] = 0.
        self._x2[:,~self.u2] = 0.
        self._x1,self._x2 = self._x1[:self.n1],self._x2[:self.n2]
    def _init_invertibile(self):
        lamblock = 1j*self.npt.empty((self.n1,self.t1,self.t1),dtype=float,**self.ckwargs)
        for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
            lamblock[:,tt1,tt2] = self.lam[tt1,tt2][0,0,0]
        self.l_chol = self.cholesky(lamblock)
    def sample(self, n_min, n_max):
        assert hasattr(self,"dd_obj"), "no discrete distribution object available to sample from"
        if self.npt==np:
            _x,x = self._sample(n_min,n_max)
        else:
            _x,x = self._sample(n_min,n_max)
            import torch 
            _x = torch.from_numpy(_x).to(device=self.ckwargs["device"])
            x = torch.from_numpy(x).to(device=self.ckwargs["device"])
        return _x,x 
    def get_full_gram_matrix(self):
        if not hasattr(self,"_x1"): self._set__x1__x2()
        gm = self._construct_full_gram_matrix(self._x1,self._x2,self.t1,self.t2,self.lbeta1s,self.lbeta2s,self.lc1s_og,self.lc2s_og)
        if self.invertible and self.noise>0:
            gm = gm+self.noise*self.npt.eye(self.size[0],dtype=float,**self.ckwargs)
        return gm
    def get_new_left_full_gram_matrix(self, new_x, new_lbetas=0, new_lcs=1.):
        new__x = self._convert_x_to__x(new_x)
        if not hasattr(self,"_x1"): self._set__x1__x2()
        new_lbetas,new_lcs,new_t,new_m = self._parse_lbetas_lcs(new_lbetas,new_lcs)
        gm = self._construct_full_gram_matrix(new__x,self._x1,new_t,self.t1,new_lbetas,self.lbeta1s,new_lcs,self.lc1s_og)
        return gm
    def multiply(self, *args, **kwargs):
        return self.__matmul__(*args, **kwargs)
    def __matmul__(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None]
        assert y.ndim==2 and y.shape[0]==(self.n2*self.t2)
        v = y.shape[1] # y is (t2*n2,v)
        yfull = y.T.reshape((v,self.t2,1,1,self.n2)) # (v,t2,n2)
        sfull = [0.]*self.t1 # (v,n1)
        for tt1 in range(self.t1):
            for tt2 in range(self.t2): 
                y = yfull[:,tt2,:,:,:]
                if self.d_u1nu2>0:
                    y = y*self.scale_null[tt1,tt2][:,:,None] # (v,m1,m2,n2) since self.scale_null is (m1,m2)
                if self.d_u2mu1>0:
                    y = y*self.k1r[tt1,tt2] # (v,m1,m2,n2) since self.k1r is (m1,m2,n2)
                if self.d_u1au2>0 or self.d_u1nu2_og==self.d:
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
                s = (self.lc1slc2s[tt1,tt2]*s).sum((1,2)) # (v,n1)
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
        y = y.T # (v,t1*n1)
        if self.t1==1:
            if self.d_u1mu2>0:
                y = y/self.k1l[0,0][0,0] # (v,self.n1) since self.k1l is (1,1,n1)
            if self.d_u1au2>0 or self.d_u1nu2_og==self.d:
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
    def _mult_check(self, y, gmatfull):
        if self.npt!=np:
            import torch
            y = torch.from_numpy(y)
        assert np.allclose(self@y[:,0],gmatfull@y[:,0],atol=1e-12)
        assert np.allclose(self@y,gmatfull@y,atol=1e-12)
    def _solve_check(self, y, gmatfull):
        if not self.invertible: return
        if self.npt==np:
            assert np.allclose(self.solve(y[:,0]),np.linalg.solve(gmatfull,y[:,0]),rtol=2.5e-2)
            assert np.allclose(self.solve(y),np.linalg.solve(gmatfull,y),rtol=2.5e-2)
        else:
            import torch 
            y = torch.from_numpy(y)
            assert np.allclose(self.solve(y[:,0]).numpy(),np.linalg.solve(gmatfull,y[:,0].numpy()),rtol=2.5e-2)
            assert np.allclose(self.solve(y).numpy(),np.linalg.solve(gmatfull,y.numpy()),rtol=2.5e-2)
    def _check(self):
        rng = np.random.Generator(np.random.SFC64(7))
        y = rng.uniform(size=(self.n2*self.t2,2))
        gmatfull = self.get_full_gram_matrix()
        self._mult_check(y,gmatfull)
        self._solve_check(y,gmatfull)

class FastGramMatrixLattice(_FastGramMatrix):
    """
    Fast Gram matrix operations using lattice points and shift invariant kernels 

    >>> import torch
    >>> n = 2**3
    >>> d = 3
    >>> dd_obj = Lattice(d,seed=7)
    >>> kernel_obj = KernelShiftInvar(d,alpha=4,torchify=True)
    >>> lbetas = [
    ...     torch.tensor([0,0,0]),
    ...     torch.tensor([1,0,0]),
    ...     torch.tensor([0,1,0]),
    ...     torch.tensor([0,0,1])]
    >>> u = torch.tensor([True,True,True])
    >>> gm = FastGramMatrixLattice(kernel_obj,dd_obj,n,n,u,u,lbetas,lbetas)
    >>> gm._check()
    >>> us = [torch.tensor([int(b) for b in np.binary_repr(i,d)],dtype=bool) for i in range(2**d)]
    >>> lbetas = [
    ...     torch.tensor([1,0,0]),
    ...     torch.tensor([[0,1,0],[0,0,1]]),
    ...     [torch.tensor([1,0,0]),torch.tensor([0,1,0])],
    ...     [torch.tensor([[1,0,0],[0,1,0],[0,0,1]]),torch.tensor([[1,0,1],[0,1,0],[0,0,0]])]
    ...     ]
    >>> num_invertible = 0
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
    ...         gm = FastGramMatrixLattice(kernel_obj,dd_obj,n1,n2,u1,u2,lbeta1s,lbeta2s,lc1s,lc2s,adaptive_noise=False)
    ...         gm._check()
    ...         gm_tall = FastGramMatrixLattice(kernel_obj,dd_obj,max(n1//2,1),max(n2//4,1),u1,u2,lbeta1s,lbeta2s,lc1s,lc2s,adaptive_noise=False)
    ...         gm_tall._check()
    ...         gm_wide = FastGramMatrixLattice(kernel_obj,dd_obj,max(n1//4,1),max(n2//2,1),u1,u2,lbeta1s,lbeta2s,lc1s,lc2s,adaptive_noise=False)
    ...         gm_wide._check()
    """
    def __init__(self, kernel_obj, dd_obj, n1, n2, u1=True, u2=True, lbeta1s=0, lbeta2s=0, lc1s=1., lc2s=1., noise=1e-8, adaptive_noise=True, _pregenerated_x__x=None):
        """
        Args:
            kernel_obj (KernelShiftInvar): shift invariant kernel
            dd_obj (Lattice): requires randomize='SHIFT' and order="NATURAL"
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
        assert isinstance(dd_obj,Lattice)
        assert dd_obj.randomize=="SHIFT"
        assert dd_obj.order=="NATURAL"
        assert dd_obj.replications==1
        assert isinstance(kernel_obj,KernelShiftInvar)
        if not kernel_obj.torchify: # numpy based 
            # if dd_obj.order=="LINEAR":
            #     # implementations from numpy
            #     self.ft = lambda x: np.fft.fft(x,norm="ortho")
            #     self.ift = lambda x: np.fft.ifft(x,norm="ortho")
            # else: # dd_obj.order=="NATURAL"
            #     # implementations from qmctoolscl (theoretically faster, practically slower)
            self.ft = fftbr 
            self.ift = ifftbr
        else: # torch 
            import torch
            # if dd_obj.order=="LINEAR":
            #     # implementations from torch
            #     self.ft = lambda x: torch.fft.fft(x,norm="ortho") 
            #     self.ift = lambda x: torch.fft.ifft(x,norm="ortho")
            # else: # dd_obj.order=="NATURAL"
            #     # custom torch implementations (theoretically faster, practically slower)
            self.ft = fftbr_torch
            self.ift = ifftbr_torch
        super(FastGramMatrixLattice,self).__init__(kernel_obj,dd_obj,n1,n2,u1,u2,lbeta1s,lbeta2s,lc1s,lc2s,noise,adaptive_noise,_pregenerated_x__x)
    def _sample(self, n_min, n_max):
        x = self.dd_obj.gen_samples(n_min=n_min,n_max=n_max)
        return x,x
    def _convert_x_to__x(self, x):
        return x
    def _convert__x_to_x(self, _x):
        return _x

class FastGramMatrixDigitalNetB2(_FastGramMatrix):
    """
    Fast Gram matrix operations using base 2 digital net points and digitally shift invariant kernels 

    >>> import torch
    >>> n = 2**3
    >>> d = 3
    >>> dd_obj = DigitalNetB2(d,t_lms=32,alpha=2,seed=7)
    >>> kernel_obj = KernelDigShiftInvar(d,alpha=4,torchify=False)
    >>> lbetas = [
    ...     np.array([0,0,0]),
    ...     np.array([1,0,0]),
    ...     np.array([0,1,0]),
    ...     np.array([0,0,1])]
    >>> u = np.array([True,True,True])
    >>> gm = FastGramMatrixDigitalNetB2(kernel_obj,dd_obj,n,n,u,u,lbetas,lbetas)
    >>> gm._check()
    >>> us = [np.array([int(b) for b in np.binary_repr(i,d)],dtype=bool) for i in range(2**d)]
    >>> lbetas = [
    ...     np.array([1,0,0]),
    ...     np.array([[0,1,0],[0,0,1]]),
    ...     [np.array([1,0,0]),np.array([0,1,0])],
    ...     [np.array([[1,0,0],[0,1,0],[0,0,1]]),np.array([[1,0,1],[0,1,0],[0,0,0]])]
    ...     ]
    >>> num_invertible = 0
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
    ...         gm = FastGramMatrixDigitalNetB2(kernel_obj,dd_obj,n1,n2,u1,u2,lbeta1s,lbeta2s,lc1s,lc2s,adaptive_noise=False)
    ...         gm._check()
    ...         gm_tall = FastGramMatrixDigitalNetB2(kernel_obj,dd_obj,max(n1//2,1),max(n2//4,1),u1,u2,lbeta1s,lbeta2s,lc1s,lc2s,adaptive_noise=False)
    ...         gm_tall._check()
    ...         gm_wide = FastGramMatrixDigitalNetB2(kernel_obj,dd_obj,max(n1//4,1),max(n2//2,1),u1,u2,lbeta1s,lbeta2s,lc1s,lc2s,adaptive_noise=False)
    ...         gm_wide._check()
    """
    def __init__(self, kernel_obj, dd_obj, n1, n2, u1=True, u2=True, lbeta1s=0, lbeta2s=0, lc1s=1., lc2s=1., noise=1e-8, adaptive_noise=True, _pregenerated_x__x=None):
        """
        Args:
            kernel_obj (KernelDigShiftInvar): digitally shift invariant kernel
            dd_obj (DigitalNetB2): requires randomize='LMS_DS' and graycode=False
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
        assert isinstance(dd_obj,DigitalNetB2)
        assert dd_obj.randomize in ["DS","LMS_DS"]
        assert dd_obj.order=="NATURAL"
        assert dd_obj.replications==1
        assert isinstance(kernel_obj,KernelDigShiftInvar)
        kernel_obj.set_t(dd_obj.t_lms)
        # FWHT is theoretically faster than FFT but practically slower
        if not kernel_obj.torchify:
            # qmctools implementation
            self.ft = fwht 
            self.ift = fwht
        else:
            # custom torch implementations 
            self.ft = fwht_torch
            self.ift = fwht_torch
        assert kernel_obj.npt==np, "FastGramMatrixDigitalNetB2 does not currently support torch as 'index_cpu' is not yet implemented for tensors of dtype torch.uint64"
        super(FastGramMatrixDigitalNetB2,self).__init__(kernel_obj,dd_obj,n1,n2,u1,u2,lbeta1s,lbeta2s,lc1s,lc2s,noise,adaptive_noise,_pregenerated_x__x)
    def _sample(self, n_min, n_max):
        xb = self.dd_obj.gen_samples(n_min=n_min,n_max=n_max,return_binary=True)
        xf = self._convert__x_to_x(xb)
        return xb,xf
    def _convert_x_to__x(self, x):
        return np.floor(x*(2**self.kernel_obj.t)).astype(np.uint64)
    def _convert__x_to_x(self, _x):
        return _x*2**(-self.kernel_obj.t)
