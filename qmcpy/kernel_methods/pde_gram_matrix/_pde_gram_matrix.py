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
        gms = [None]*self.nr
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
        return [zs.reshape((self.gms[i,0].t1,-1)) for i,zs in enumerate(np.split(z,self.bs_cumsum))]
    def compose(self, lzs):
        return self.npt.hstack([zs.flatten() for zs in lzs])
    def decompose_eqns(self, z):
        return np.split(z,self.n_cumsum)
    def _loss(self, F, y):
        diff = y-F
        loss = self.npt.sqrt((diff**2).mean())
        return loss
    def gauss_newton_relaxed(self, pde_lhs, pde_rhs, maxiter=10, pcg_rtol=None, pcg_atol=None, pcg_maxiter=None, pcg_precond=False, verbose=True):
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
        losses = self.npt.nan*self.npt.zeros(maxiter+1)
        rbackward_norms = [None]*maxiter
        times = [None]*maxiter
        xs = self._get_xs()
        y = pde_rhs_wrap(xs) 
        zt = torch.zeros(self.length,dtype=torch.float64,requires_grad=True)
        Ft = pde_lhs_wrap(zt)
        F = Ft.detach().numpy() if self.npt==np else Ft.detach()
        losses[0] = self._loss(F,y)
        if verbose: print("\titeration i/%d: i = "%maxiter,end='',flush=True)
        for i in range(maxiter):
            if verbose: print("%d, "%(i+1),end='',flush=True)
            Fpt = torch.autograd.grad(Ft,zt,grad_outputs=torch.ones_like(Ft))[0]
            z = zt.detach().numpy() if self.npt==np else zt.detach()
            Fp = Fpt.detach().numpy() if self.npt==np else Fpt.detach()
            Fpd = self.decompose(Fp)
            def mult_Fp(a):
                ad = self.decompose(a) 
                bd = [(Fpd[i]*ad[i]).sum(0) for i in range(self.nr)]
                b = self.compose(bd) 
                return b
            def mult_tFp(b):
                bd = self.decompose_eqns(b) 
                ad = [Fpd[i]*bd[i] for i in range(self.nr)]
                a = self.compose(ad) 
                return a
            def multiply(gamma):
                t1 = mult_tFp(gamma) 
                t2 = self@t1 
                t3 = mult_Fp(t2)
                return t3
            Fpz = mult_Fp(z)
            diff = y-F+Fpz
            gamma,rbackward_norms[i],times[i] = pcg(matmul=multiply,b=diff,x0=None,rtol=pcg_rtol,atol=pcg_atol,maxiter=pcg_maxiter,precond_solve=pcg_precond,ref_solver=False, npt=self.npt, ckwargs=self.ckwargs)
            #z_old = z
            tFpgamma = mult_tFp(gamma)
            z = self@tFpgamma
            zt = torch.from_numpy(z).requires_grad_() if self.npt==np else z.requires_grad_()
            Ft = pde_lhs_wrap(zt) 
            F = Ft.detach().numpy() if self.npt==np else Ft.detach()
            losses[i+1] = self._loss(F,y)
        rbackward_norms,times = rbackward_norms[:(i+1)],times[:(i+1)]
        max_pcg_iters = max([len(rbackward_norm) for rbackward_norm in rbackward_norms])
        rbackward_norms_mat = self.npt.nan*self.npt.empty((i+1,max_pcg_iters))
        times_mat = self.npt.nan*self.npt.empty((i+1,max_pcg_iters))
        for l in range(i+1):
            rbackward_norms_mat[l,:len(rbackward_norms[l])] = rbackward_norms[l]
            times_mat[l,:len(times[l])] = times[l] 
        if verbose: print()
        z = zt.detach().numpy() if self.npt==np else zt.detach() 
        return z,losses[:(i+2)],rbackward_norms_mat,times_mat
