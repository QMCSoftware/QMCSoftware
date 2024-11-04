from ..discrete_distribution import Lattice
from .kernels import KernelShiftInvar
import numpy as np
import scipy.linalg
import itertools

class GramMatrix(object):
    """ Gram Matrix 

    >>> d = 3
    >>> n = 2**3
    >>> lbetas = [
    ...     np.array([1,0,0]),
    ...     np.array([[0,1,0],[0,0,1]]),
    ...     [np.array([1,0,0]),np.array([0,1,0])],
    ...     [np.array([[1,0,0],[0,1,0],[0,0,1]]),np.array([[1,0,1],[0,1,0],[0,0,0]])]
    ...     ]
    >>> x = Lattice(d,seed=7).gen_samples(n)
    >>> kernel_obj = KernelShiftInvar(d)
    >>> for n1,n2 in itertools.product([n//2,n],[n//2,n]):
    ...     for ib1,ib2 in itertools.product(range(len(lbetas)),range(len(lbetas))):
    ...         x1 = x[:n1]
    ...         x2 = x[:n2]
    ...         lbeta1s = lbetas[ib1] 
    ...         lbeta2s = lbetas[ib2]
    ...         gm = GramMatrix(x1,x2,kernel_obj,lbeta1s,lbeta2s)
    ...         gm._check()
    """
    def __init__(self, x1, x2, kernel_obj, lbeta1s=0, lbeta2s=0, lc1s=1., lc2s=1., noise=1e-8):
        """
        Args:
            x1 (np.ndarray or torch.Tensor): n1 left locations
            x2 (np.ndarray or torch.Tensor): n2 right locations
            kernel_obj (kernel): kernel
            lbeta1s (list of either np.ndarray or torch.Tensor): list of (m1[l],d) arrays of first derivative orders
            lbeta1s (list of either np.ndarray or torch.Tensor): list of (m2[l],d) arrays of first derivative orders
            c1s (list of either np.ndarray or torch.Tensor): list of length m1[l] vectors of derivative coefficients 
            c2s (list of either np.ndarray or torch.Tensor): list of length m2[l] vector of derivative coefficients
            noise (float): nugget term 
        """
        self.kernel_obj = kernel_obj
        self.d = self.kernel_obj.d
        self.npt = self.kernel_obj.npt
        self.torchify = self.kernel_obj.torchify
        self.n1 = len(x1) 
        self.n2 = len(x2) 
        assert x1.shape==(self.n1,self.d) and x2.shape==(self.n2,self.d)
        self.noise = noise
        assert isinstance(self.noise,float) and self.noise>0.
        if isinstance(lbeta1s,int): lbeta1s = [lbeta1s*self.npt.ones((1,self.d),dtype=int)]
        elif not isinstance(lbeta1s,list): lbeta1s = [self.npt.atleast_2d(lbeta1s)]
        if isinstance(lbeta2s,int): lbeta2s = [lbeta2s*self.npt.ones((1,self.d),dtype=int)]
        elif not isinstance(lbeta2s,list): lbeta2s = [self.npt.atleast_2d(lbeta2s)]
        self.lbeta1s = [self.npt.atleast_2d(beta1s) for beta1s in lbeta1s]
        self.lbeta2s = [self.npt.atleast_2d(beta2s) for beta2s in lbeta2s]
        self.t1 = len(self.lbeta1s)
        self.t2 = len(self.lbeta2s)
        self.m1 = np.array([len(beta1s) for beta1s in self.lbeta1s],dtype=int)
        self.m2 = np.array([len(beta2s) for beta2s in self.lbeta2s],dtype=int)
        assert isinstance(self.lbeta1s,list) and all(self.lbeta1s[tt1].shape==(self.m1[tt1],self.d) for tt1 in range(self.t1))
        assert isinstance(self.lbeta2s,list) and all(self.lbeta2s[tt2].shape==(self.m2[tt2],self.d) for tt2 in range(self.t2))
        if isinstance(lc1s,float): lc1s = [lc1s*self.npt.ones(self.m1[tt1]) for tt1 in range(self.t1)]
        elif not isinstance(lc1s,list): lc1s = [lc1s]
        if isinstance(lc2s,float): lc2s = [lc2s*self.npt.ones(self.m2[tt2]) for tt2 in range(self.t2)]
        elif not isinstance(lc2s,list): lc2s = [lc2s]
        self.lc1s = [self.npt.atleast_1d(c1s) for c1s in lc1s] 
        self.lc2s = [self.npt.atleast_1d(c2s) for c2s in lc2s]
        self.lc1s_og = [self.npt.atleast_1d(c1s) for c1s in lc1s] 
        self.lc2s_og = [self.npt.atleast_1d(c2s) for c2s in lc2s]
        assert isinstance(self.lc1s,list) and all(self.lc1s[tt1].shape==(self.m1[tt1],) for tt1 in range(self.t1))
        assert isinstance(self.lc2s,list) and all(self.lc2s[tt2].shape==(self.m2[tt2],) for tt2 in range(self.t2))
        gms = np.empty((self.t1,self.t2),dtype=object) 
        for tt1,tt2 in itertools.product(range(self.t1),range(self.t2)):
            gms[tt1,tt2] = kernel_obj(x1,x2,lbeta1s[tt1],lbeta2s[tt2],lc1s[tt1],lc2s[tt2])
        self.gm = self.npt.vstack([self.npt.hstack([gms[tt1,tt2] for tt2 in range(self.t2)]) for tt1 in range(self.t1)])
        invertible_conds = [
            ( x1.ctypes.data==x2.ctypes.data, "x1 and x2 must point to the same object"),
            ( self.n1==self.n2, "require square matrices"),
            ( self.t1==self.t2 and all((self.lbeta1s[tt1]==self.lbeta2s[tt1]).all() and (self.lc1s[tt1]==self.lc2s[tt1]).all() for tt1 in range(self.t1)), "require lbeta1s=lbeta2s and lc1s=lc2s"),
            ]
        self.invertible = all(cond for cond,error_msg in invertible_conds)
        self.invertible_error_msg = "\n\n"+"\n\n".join(error_msg for cond,error_msg in invertible_conds if not cond)
        if self.invertible:
            try:
                self.l_chol = self.npt.linalg.cholesky(self.gm+self.noise*self.npt.eye(self.n1*self.t1))
            except:
                raise Exception("Cholesky not positive definite, try increasing the noise")
            if self.torchify:
                import torch 
                self.cho_solve = lambda l,b: torch.cholesky_solve(b,l,upper=False)
            else:
                self.cho_solve = lambda l,b: scipy.linalg.cho_solve((l,True),b)
        self.size = (self.n1*self.t1,self.n2*self.t2)
    def get_full_gram_matrix(self):
        return self.gm
    def multiply(self, *args, **kwargs):
        return self.__matmul__(*args, **kwargs)
    def __matmul__(self, y):
        return self.gm@y
    def solve(self, y):
        assert self.invertible, self.invertible_error_msg
        return self.cho_solve(self.l_chol,y)
    def _check(self):
        if not self.invertible: return
        rng = np.random.Generator(np.random.SFC64(7))
        y = rng.uniform(size=(self.n2*self.t2,2))
        assert np.allclose(self.solve(y[:,0]),np.linalg.solve(self.gm,y[:,0]),rtol=2.5e-2)
        assert np.allclose(self.solve(y),np.linalg.solve(self.gm,y),rtol=2.5e-2)
        
