from ._gram_matrix import _GramMatrix
from ...discrete_distribution import IIDStdUniform
from ..kernel import KernelGaussian
from ..pcg import pcg,BlockPrecond
import numpy as np
import itertools

class GramMatrix(_GramMatrix):
    """ Gram Matrix 

    >>> import torch 

    >>> d = 3
    >>> n = 2**3
    >>> dd_obj = IIDStdUniform(d,seed=7)
    >>> x = torch.from_numpy(dd_obj.gen_samples(n))
    >>> kernel_obj = KernelGaussian(d,torchify=True)
    >>> lbetas = [
    ...     torch.tensor([0,0,0]),
    ...     torch.tensor([0,2,0])]
    >>> gm = GramMatrix(kernel_obj,x,x,lbetas,lbetas)
    >>> gm._check()
    >>> y = torch.from_numpy(dd_obj.rng.uniform(size=gm.t1*n))
    >>> gm@y
    tensor([-0.3561,  0.0768,  0.0835, -0.3175, -0.0122,  0.4073,  0.2060,  0.0110,
            10.1762,  6.1081,  6.6691,  9.1797,  7.4724,  3.9216,  4.8994,  6.7917],
           dtype=torch.float64)
    >>> v = gm.solve(y)
    >>> v
    tensor([-1.9905,  3.3829, -8.1582,  0.9399,  1.8549,  7.6392,  0.0553, -1.7623,
            -0.3368,  0.2528,  0.3169,  0.3104,  0.2775,  0.0470,  0.0763, -0.7187],
           dtype=torch.float64)
    >>> precond = BlockPrecond(gm.full_mat,n_cumsum=[0,len(v)//2,len(v)])
    >>> vhat,data = pcg(gm.full_mat,y,precond,ref_sol=v)
    >>> data["rforward_norms"]
    tensor([1.0000e+00, 3.6825e-01, 2.1724e-01, 2.1260e-01, 1.2399e-02, 7.2391e-03,
            4.6767e-03, 4.5761e-03, 7.3053e-04, 5.6551e-04, 2.2098e-04, 1.4451e-04,
            4.0510e-06, 3.2320e-06, 6.2216e-07, 3.8075e-07, 7.2468e-15],
           dtype=torch.float64)
    >>> print(precond.info_str(gm.full_mat))
    K(A)           K(P)           K(P)/K(A)      
    2.1e+04        3.2e+03        1.5e-01        
    >>> vhat
    tensor([-1.9905,  3.3829, -8.1582,  0.9399,  1.8549,  7.6392,  0.0553, -1.7623,
            -0.3368,  0.2528,  0.3169,  0.3104,  0.2775,  0.0470,  0.0763, -0.7187],
           dtype=torch.float64)
           
    >>> lbetas = [
    ...     torch.tensor([1,0,0]),
    ...     torch.tensor([[0,1,0],[0,0,1]]),
    ...     [torch.tensor([1,0,0]),torch.tensor([0,1,0])],
    ...     [torch.tensor([[1,0,0],[0,1,0],[0,0,1]]),torch.tensor([[1,0,1],[0,1,0],[0,0,0]])]
    ...     ]
    >>> for n1,n2 in itertools.product([n//2,n],[n//2,n]):
    ...     for ib1,ib2 in itertools.product(range(len(lbetas)),range(len(lbetas))):
    ...         x1 = x[:n1]
    ...         x2 = x[:n2]
    ...         lbeta1s = lbetas[ib1] 
    ...         lbeta2s = lbetas[ib2]
    ...         gm = GramMatrix(kernel_obj,x1,x2,lbeta1s,lbeta2s,adaptive_noise=False)
    ...         gm._check()
    """
    def __init__(self, kernel_obj, x1, x2, lbeta1s=0, lbeta2s=0, lc1s=1., lc2s=1., noise=1e-8, adaptive_noise=True):
        """
        Args:
            kernel_obj (kernel): kernel
            x1 (np.ndarray or torch.Tensor): n1 left locations
            x2 (np.ndarray or torch.Tensor): n2 right locations
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
        self.full_mat = self._construct_full_gram_matrix(self.x1,self.x2,self.t1,self.t2,self.lbeta1s,self.lbeta2s,self.lc1s_og,self.lc2s_og)
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
                traces = self.full_mat.diagonal()[::self.n1]
                trace_ratios = traces/traces[0]
                self.full_mat += noise*self.npt.diag((self.npt.ones((self.n1,len(traces)))*trace_ratios).T.flatten())
            else:
                self.full_mat += self.noise*self.npt.eye(self.size[0],dtype=float,**self.ckwargs)
        self._l_chol = None 
    @property
    def l_chol(self):
        if self._l_chol is None: 
            self._l_chol = self.cholesky(self.full_mat)
        return self._l_chol
    def get_new_left_full_gram_matrix(self, new_x, new_lbetas=0, new_lcs=1.):
        new_lbetas,new_lcs,new_t,new_m = self._parse_lbetas_lcs(new_lbetas,new_lcs)
        gm = self._construct_full_gram_matrix(self.x1,new_x,self.t1,new_t,self.lbeta1s,new_lbetas,self.lc1s_og,new_lcs).T
        return gm
    def _init_invertibile(self):
        self.l_chol = self.cholesky(self.full_mat)
    def multiply(self, *args, **kwargs):
        return self.__matmul__(*args, **kwargs)
    def __matmul__(self, y):
        return self.full_mat@y
    def solve(self, y):
        assert self.invertible, self.invertible_error_msg
        return self.cho_solve(self.l_chol,y)
    def _check(self):
        if not self.invertible: return
        rng = np.random.Generator(np.random.SFC64(7))
        y = rng.uniform(size=(self.n2*self.t2,2))
        if self.npt==np:
            assert np.allclose(self.solve(y[:,0]),np.linalg.solve(self.full_mat,y[:,0]),rtol=2.5e-2)
            assert np.allclose(self.solve(y),np.linalg.solve(self.full_mat,y),rtol=2.5e-2)
        else:
            import torch
            y = torch.from_numpy(y)
            assert np.allclose(self.solve(y[:,0]).numpy(),np.linalg.solve(self.full_mat.numpy(),y[:,0]),rtol=2.5e-2)
            assert np.allclose(self.solve(y).numpy(),np.linalg.solve(self.full_mat.numpy(),y),rtol=2.5e-2)