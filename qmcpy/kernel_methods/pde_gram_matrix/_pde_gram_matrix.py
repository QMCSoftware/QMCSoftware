import numpy as np 
import time 
from ..util import pcg

class _PDEGramMatrix(object):
    def __init__(self, kernel_obj, llbetas, llcs):
        self.kernel_obj = kernel_obj
        self.npt = self.kernel_obj.npt
        self.ckwargs = self.kernel_obj.ckwargs
        self.llbetas,self.llcs,_inferred_nr = self._parse_llbetas_llcs(llbetas,llcs)
        assert _inferred_nr==self.nr
    def _parse_llbetas_llcs(self, llbetas, llcs):
        assert isinstance(llbetas,list) and all(isinstance(lbetas,list) for lbetas in llbetas)
        nr = len(llbetas)
        assert isinstance(llcs,list) and all(isinstance(lcs,list) for lcs in llcs) and len(llcs)==nr
        return llbetas,llcs,nr
    def get_new_left_full_gram_matrix(self, new_x, new_lbetas, new_lcs):
        gms = np.empty(self.nr,dtype=object)
        for i in range(self.nr):
            gms[i] = self.gms[i,0].get_new_left_full_gram_matrix(new_x,new_lbetas,new_lcs)
        return self.npt.hstack(gms) 
    def _mult_check(self):
        rng = np.random.Generator(np.random.SFC64(7))
        y = rng.uniform(size=(self.length,2))
        gmatfull = self.get_full_gram_matrix()
        assert np.allclose(self@y[:,0],gmatfull@y[:,0],atol=1e-12)
        assert np.allclose(self@y,gmatfull@y,atol=1e-12)
    def multiply(self, *args, **kwargs):
        return self.__matmul__(*args, **kwargs)
    def pcg(self, b, x0=None, rtol=1e-8, atol=0., maxiter=None, precond=False, ref_sol=False):
        assert precond is True or precond is False or isinstance(precond,str) 
        if precond is True:
            precond = self.precond_solve
        elif isinstance(precond,str):
            precond = getattr(self,precond)
        assert isinstance(ref_sol,bool)
        if ref_sol is True: 
            ref_sol = self._solve
        return pcg(self.__matmul__,b,x0,rtol,atol,maxiter,precond_solve=precond,ref_solver=ref_sol,npt=self.npt,ckwargs=self.ckwargs)
    def _solve(self, y):
        if not hasattr(self,"l_chol"): 
            self._init_invertibile()
        return self.cho_solve(self.l_chol,y)
    def decompose(self, z):
        return [np.split(zs,self.gms[i,0].t1) for i,zs in enumerate(np.split(z,self.bs_cumsum))]
    def gauss_newton_relaxed(self, pde_lhs, pde_rhs, maxiter=10, relaxation=1e-8, pcg_rtol=None, pcg_atol=None, pcg_maxiter=None, pcg_precond=False, verbose=True):
        def pde_lhs_wrap(z):
            zd = self.decompose(z)
            y_lhss = pde_lhs(*zd)
            y_lhs = torch.hstack(y_lhss)
            return y_lhs 
        def pde_rhs_wrap(xs):
            y_rhss = pde_rhs(*xs)
            y_rhs = self.npt.hstack(y_rhss) 
            return y_rhs
        try:
            import torch
        except: 
            raise Exception("gauss_newton_relaxed requires torch for automatic differentiation through pde_lhs")
        assert pcg_precond is True or pcg_precond is False or isinstance(pcg_precond,str) 
        if pcg_precond is True:
            pcg_precond = self.precond_solve
        elif isinstance(pcg_precond,str):
            pcg_precond = getattr(self,pcg_precond)
        xs = self._get_xs()
        y = pde_rhs_wrap(xs) 
        zt = torch.zeros(self.length,dtype=torch.float64)
        if verbose: print("\titeration i/%d: i = "%maxiter,end='',flush=True)
        for i in range(maxiter):
            if verbose: print("%d, "%(i+1),end='',flush=True)
            Ft = pde_lhs_wrap(zt) 
            Fpt = torch.autograd.functional.jacobian(pde_lhs_wrap,zt)
            if self.npt==np:
                z,F,Fp = zt.numpy(),Ft.numpy(),Fpt.numpy()
            else:
                z,F,Fp = zt,Ft,Fpt
            Fpsq = Fp.T@Fp
            def multiply(delta):
                return self@(Fpsq@delta)+relaxation*delta
            b = self@(Fp.T@(y-F))-relaxation*z
            delta,rbackward_norms,times = pcg(matmul=multiply,b=b,x0=None,rtol=pcg_rtol,atol=pcg_atol,maxiter=pcg_maxiter,precond_solve=pcg_precond,ref_solver=False, npt=self.npt, ckwargs=self.ckwargs)
            deltat = torch.from_numpy(delta) if self.npt==np else delta
            zt = zt+deltat
        if verbose: print()
        return zt.numpy() if self.npt==np else zt  


