from ._gram_matrix import _GramMatrix
from ...discrete_distribution import IIDStdUniform
from ..kernel import KernelShiftInvar
import numpy as np
import itertools

class GramMatrix(_GramMatrix):
    """ Gram Matrix 

    >>> d = 3
    >>> n = 2**3
    >>> lbetas = [
    ...     np.array([1,0,0]),
    ...     np.array([[0,1,0],[0,0,1]]),
    ...     [np.array([1,0,0]),np.array([0,1,0])],
    ...     [np.array([[1,0,0],[0,1,0],[0,0,1]]),np.array([[1,0,1],[0,1,0],[0,0,0]])]
    ...     ]
    >>> x = IIDStdUniform(d,seed=7).gen_samples(n)
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
    def __init__(self, x1, x2, kernel_obj, lbeta1s=0, lbeta2s=0, lc1s=1., lc2s=1., noise=1e-8, adaptive_noise=True):
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
            adaptive_noise (bool): if True, use the adative noise scheme described in Appendix A of 
                Chen, Yifan, et al. "Solving and learning nonlinear PDEs with Gaussian processes." 
                Journal of Computational Physics 447 (2021): 110668.
        """
        super(GramMatrix,self).__init__(kernel_obj,noise,lbeta1s,lbeta2s,lc1s,lc2s)
        self.x1,self.x2 = x1,x2
        self.n1 = len(self.x1) 
        self.n2 = len(self.x2) 
        assert self.x1.shape==(self.n1,self.d) and self.x2.shape==(self.n2,self.d)
        self.gm = self._construct_full_gram_matrix(self.x1,self.x2,self.t1,self.t2,self.lbeta1s,self.lbeta2s,self.lc1s_og,self.lc2s_og)
        self.size = (self.n1*self.t1,self.n2*self.t2)
        invertible_conds = [
            ( self.get_ptr(self.x1)==self.get_ptr(self.x2), "x1 and x2 must point to the same object"),
            ( self.n1==self.n2, "require square matrices"),
            ( self.t1==self.t2 and all((self.lbeta1s[tt1]==self.lbeta2s[tt1]).all() and (self.lc1s[tt1]==self.lc2s[tt1]).all() for tt1 in range(self.t1)), "require lbeta1s=lbeta2s and lc1s=lc2s"),
            ]  
        super(GramMatrix,self)._set_invertible_conds(invertible_conds)
        if self.invertible:
            if adaptive_noise:
                assert (self.lbeta1s[0]==0.).all() and (self.lbeta1s[0].shape==(1,self.d)) and (self.lc1s_og[0]==1.).all() and (self.lc1s_og[0].shape==(1,))
                traces = self.gm.diagonal()[::self.n1]
                trace_ratios = traces/traces[0]
                self.gm += noise*self.npt.diag((self.npt.ones((self.n1,len(traces)))*trace_ratios).T.flatten())
            else:
                self.gm += self.noise*self.npt.eye(self.size[0],dtype=float,**self.ckwargs)
    def get_full_gram_matrix(self):
        return self.gm.copy()
    def get_new_left_full_gram_matrix(self, new_x, new_lbetas, new_lcs):
        new_lbetas,new_lcs,new_t,new_m = self._parse_lbetas_lcs(new_lbetas,new_lcs)
        gm = self._construct_full_gram_matrix(self.x1,new_x,self.t1,new_t,self.lbeta1s,new_lbetas,self.lc1s_og,new_lcs).T
        return gm
    def _init_invertibile(self):
        self.l_chol = self.cholesky(self.gm)
    def multiply(self, *args, **kwargs):
        return self.__matmul__(*args, **kwargs)
    def __matmul__(self, y):
        return self.gm@y
    def solve(self, y):
        assert self.invertible, self.invertible_error_msg
        if not hasattr(self,"l_chol"): 
            self._init_invertibile()
        return self.cho_solve(self.l_chol,y)
    def _check(self):
        if not self.invertible: return
        rng = np.random.Generator(np.random.SFC64(7))
        y = rng.uniform(size=(self.n2*self.t2,2))
        assert np.allclose(self.solve(y[:,0]),np.linalg.solve(self.gm,y[:,0]),rtol=2.5e-2)
        assert np.allclose(self.solve(y),np.linalg.solve(self.gm,y),rtol=2.5e-2)
