from ..util import _get_npt
from .partial_pivoted_chol import ppchol,solve_ppchol
import numpy as np 
import scipy.linalg 

class _BasePrecond(object):
    def __init__(self, *args, **kwargs): 
        pass
    def solve(self, x):
        raise NotImplementedError()
    def _log_header(self):
        return "%-15s%-15s%-15s"%("K(A)","K(P)","K(P)/K(A)")
    def _log(self, mat):
        npt = _get_npt(mat)
        assert mat.ndim==2
        cond_mat = npt.linalg.cond(mat,p="fro")
        p = self.solve(mat) 
        cond_p = npt.linalg.cond(p,p="fro")
        return "%-15.1e%-15.1e%-15.1e"%(cond_mat,cond_p,cond_p/cond_mat)
    def info_str(self, mat, spaces=0):
        s1 = self._log_header()
        s2 = self._log(mat) 
        return " "*spaces+s1+"\n"+" "*spaces+s2

class IdentityPrecond(_BasePrecond):
    def __init__(self):
        pass 
    def solve(self, x):
        return x

class PPCholPrecond(_BasePrecond):
    def __init__(self, mat, ddiag_const=1e-8, rank=None, rtol=None, atol=None):
        npt = _get_npt(mat)
        self.ddiag = ddiag_const*npt.ones_like(mat.diagonal())
        self.Lk = ppchol(mat-npt.diag(self.ddiag),rank=rank,rtol=rtol,atol=atol)
        k = self.Lk.shape[1]
        self.pL = npt.linalg.cholesky(npt.eye(k,dtype=float)+(self.Lk.T/self.ddiag)@self.Lk)
    def solve(self, x):
        return solve_ppchol(x,self.Lk,self.pL,self.ddiag)
    def _log_header(self):
        s = super()._log_header()
        s_new = "%-15s"%"Lk.shape"
        return s+s_new 
    def _log(self,mat):
        s = super()._log(mat)
        s_new = "%-15s"%str(tuple(self.Lk.shape))
        return s+s_new

class JacobiPrecond(_BasePrecond):
    def __init__(self, mat):
        self.ddiag = mat.diagonal()
    def solve(self, x):
        dimx = x.ndim
        assert dimx==1 or dimx==2
        if dimx==1: x = x[:,None]
        y = x/self.ddiag
        return y[:,0] if dimx==1 else y

class SSORPrecond(_BasePrecond):
    def __init__(self, mat, omega=1):
        assert np.isscalar(omega) and 0<omega<2
        self.omega = omega 
        self.ddiag_over_omega = mat.diagonal()/self.omega
        self.npt = _get_npt(mat)
        self.L_ssor = self.npt.tril(mat,-1)
        if self.npt==np:
            np.fill_diagonal(self.L_ssor,self.ddiag_over_omega)
            self.solve_triangular = lambda L,x,upper: scipy.linalg.solve_triangular(L,x,lower=(not upper))
        else:
            import torch
            self.L_ssor.diagonal().copy_(self.ddiag_over_omega)
            self.solve_triangular = lambda L,x,upper: torch.linalg.solve_triangular(L,x,upper=upper)
    def solve(self, x):
        dimx = x.ndim
        assert dimx==1 or dimx==2
        if dimx==1: x = x[:,None]
        t1 = self.solve_triangular(self.L_ssor,x,upper=False)
        t2 = t1*self.ddiag_over_omega[:,None]
        t3 = self.solve_triangular(self.L_ssor.T,t2,upper=True)
        t4 = (2-self.omega)*t3
        return t4[:,0] if dimx==1 else t4

class BlockPrecond(_BasePrecond):
    def __init__(self, mat, n_cumsum):
        self.npt = _get_npt(mat)
        self.n_cumsum = n_cumsum 
        self.nr = len(self.n_cumsum)-1
        self.L_chol_blocks = np.empty(self.nr,dtype=object) 
        for si in range(self.nr):
            sl,sh = self.n_cumsum[si],self.n_cumsum[si+1]
            self.L_chol_blocks[si] = self.npt.linalg.cholesky(mat[sl:sh,sl:sh])
        if self.npt==np:
            self.cho_solve = lambda l,b: scipy.linalg.cho_solve((l,True),b)
        else:
            import torch 
            self.cho_solve = lambda l,b: torch.cholesky_solve(b.reshape(len(b),-1),l,upper=False).reshape(b.shape)
    def solve(self, x):
        dimx = x.ndim
        assert dimx==1 or dimx==2
        if dimx==1: x = x[:,None]
        y = self.npt.empty_like(x)
        for si in range(self.nr):
            sl,sh = self.n_cumsum[si],self.n_cumsum[si+1]
            y[sl:sh] = self.cho_solve(self.L_chol_blocks[si],x[sl:sh])
        return y[:,0] if dimx==1 else y

class FastBlockPrecond(_BasePrecond):
    def __init__(self, fpde_gm):
        assert type(fpde_gm).__name__=="FastPDEGramMatrix"
        self.fpde_gm = fpde_gm
    def solve(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None] # (l,v)
        assert y.ndim==2 and y.shape[0]==(self.fpde_gm.length)
        ysplit = np.split(y,self.fpde_gm.bs_cumsum[1:-1]) # (li,v)
        ssplit = [self.fpde_gm.gms[i,i].solve(ysplit[i]) for i in range(self.fpde_gm.nr)]
        s = self.fpde_gm.npt.vstack(ssplit)  # (l,v)
        return s[:,0] if yogndim==1 else s
