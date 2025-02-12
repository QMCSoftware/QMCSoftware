import numpy as np 
import scipy.linalg
import itertools
from ..pcg_module import pcg

class _GramMatrix(object):
    def __init__(self, kernel_obj, noise, lbeta1s, lbeta2s, lc1s, lc2s):
        self.kernel_obj = kernel_obj
        self.d = self.kernel_obj.d
        self.npt = self.kernel_obj.npt
        self.ckwargs = self.kernel_obj.ckwargs
        self.torchify = self.kernel_obj.torchify
        self.noise = noise
        self.lbeta1s,self.lc1s,self.t1,self.m1 = self._parse_lbetas_lcs(lbeta1s,lc1s)
        self.lbeta2s,self.lc2s,self.t2,self.m2 = self._parse_lbetas_lcs(lbeta2s,lc2s)
        if self.torchify:
            import torch 
            self.transpose_func = lambda x,dims: torch.permute(x,dims)
            self.clone = lambda x: x.clone()
            self.get_ptr = lambda x: x.data_ptr()
        else:
            self.transpose_func = lambda x,dims: x.transpose(*dims)
            self.clone = lambda x: x.copy()
            self.get_ptr = lambda x: x.ctypes.data
        self.lc1s_og = [self.clone(c1s) for c1s in self.lc1s] 
        self.lc2s_og = [self.clone(c2s) for c2s in self.lc2s]
    def cho_solve(self, l, b):
        bis1d = b.ndim==1
        if bis1d:
            b = b[:,None]
        if self.npt==np:
            vcs = np.vectorize(lambda l,b: scipy.linalg.cho_solve((l,True),b),signature="(m,m),(m,k)->(m,k)")
            v = vcs(l,b)
        else:
            import torch 
            v = torch.cholesky_solve(b,l,upper=False)
        if bis1d:
            v = v[:,0]
        return v
    def _parse_lbetas_lcs(self, lbetas, lcs):
        if isinstance(lbetas,int): lbetas = [lbetas*self.npt.ones((1,self.d),dtype=int,**self.ckwargs)]
        elif not isinstance(lbetas,list): lbetas = [self.npt.atleast_2d(lbetas)]
        lbetas = [self.npt.atleast_2d(beta1s) for beta1s in lbetas]
        t = len(lbetas)
        m = np.array([len(beta1s) for beta1s in lbetas],dtype=int)
        assert isinstance(lbetas,list) and all(lbetas[tt].shape==(m[tt],self.d) for tt in range(t))
        if np.isscalar(lcs): lcs = [lcs*self.npt.ones(m[tt],dtype=float,**self.ckwargs) for tt in range(t)]
        elif not isinstance(lcs,list): lcs = [lcs]
        lcs = [self.npt.atleast_1d(c1s) for c1s in lcs] 
        assert isinstance(lcs,list) and all(lcs[tt].shape==(m[tt],) for tt in range(t))
        return lbetas,lcs,t,m
    def cholesky(self, mat):
        try:
            l_chol = self.npt.linalg.cholesky(mat)
        except:
            raise Exception("Cholesky not positive definite, try increasing the noise")
        return l_chol
    def _set_invertible_conds(self, invertible_conds):
        self.invertible = all(cond for cond,error_msg in invertible_conds)
        self.invertible_error_msg = "\n\n"+"\n\n".join(error_msg for cond,error_msg in invertible_conds if not cond)
    def _construct_full_gram_matrix(self, x1, x2, t1, t2, lbeta1s, lbeta2s, lc1s, lc2s):
        gms = np.empty((t1,t2),dtype=object) 
        for tt1,tt2 in itertools.product(range(t1),range(t2)):
            gms[tt1,tt2] = self.kernel_obj(x1,x2,lbeta1s[tt1],lbeta2s[tt2],lc1s[tt1],lc2s[tt2])
        gm = self.npt.vstack([self.npt.hstack([gms[tt1,tt2] for tt2 in range(t2)]) for tt1 in range(t1)])
        return gm
