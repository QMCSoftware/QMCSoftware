import torch.linalg
import scipy.linalg
from ..util import pcg,ppchol,solve_ppchol
import numpy as np 
import time 

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
    def get_new_left_full_gram_matrix(self, new_x, new_lbetas=0, new_lcs=1.):
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
    def _init_precond_solve(self):
        pass
    def pcg(self, b, x0=None, rtol=1e-8, atol=0., maxiter=None, precond=False, ref_sol=False):
        assert precond is True or precond is False or isinstance(precond,str) 
        if precond is True:
            self._init_precond_solve()
            precond = self.precond_solve
        assert isinstance(ref_sol,bool)
        ref_sol = self._solve(b) if ref_sol else None
        return pcg(self.__matmul__,b,x0,rtol,atol,maxiter,precond_solve=precond,ref_sol=ref_sol,npt=self.npt,ckwargs=self.ckwargs)
    def _solve(self, y):
        if not hasattr(self,"l_chol"): 
            self._init_invertibile()
        return self.cho_solve(self.l_chol,y)
    def _loss(self, F, y):
        diff = y-F
        loss = self.npt.sqrt((diff**2).mean())
        return loss
    def pde_opt_gauss_newton(self, pde_lhs, pde_rhs, maxiter=10, relaxation=0., solver="CHOL", verbose=1,
            pcg_rtol=None, pcg_atol=None, pcg_maxiter=None, pcg_beta_method=None,
            ppchol_rank=None, ppchol_rtol=None, ppchol_atol=None,
            ssor_omega=1):
        def pde_lhs_wrap(z):
            zd = [zs.reshape((self.tvec[j],-1)) for j,zs in enumerate(np.split(z,self.bs_cumsum[1:-1]))]
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
            raise Exception("pde_opt_gauss_newton requires torch for automatic differentiation through pde_lhs")
        assert isinstance(solver,str)
        solver = solver.upper() 
        assert solver in ["CHOL","CG","PCG-PPCHOL","PCG-JACOBI","PCG-SSOR","PCG-BLOCKED"]
        fast_flag = not hasattr(self,"gm") # isinstance(self,FastPDEGramMatrix)
        losses = self.npt.nan*self.npt.ones(maxiter+1)
        rbackward_norms = [self.npt.nan*self.npt.ones(1)]*maxiter
        times = [self.npt.nan*self.npt.ones(1)]*maxiter
        if verbose: print("\t%-20s%-15s"%("iter (%d max)"%maxiter,"loss"))
        xs = self.get_xs()
        y = pde_rhs_wrap(xs) 
        zhist = torch.zeros((maxiter+1,self.length),dtype=torch.float64)
        zt = torch.zeros(self.length,dtype=torch.float64,requires_grad=True)
        Ft = pde_lhs_wrap(zt)
        F = Ft.detach().numpy() if self.npt==np else Ft.detach()
        z = zt.detach().numpy() if self.npt==np else zt.detach()
        zhist[0,:] = z
        losses[0] = self._loss(F,y)
        loss_best = losses[0] 
        z_best = z
        gm = self.get_full_gram_matrix()
        Theta = torch.from_numpy(gm) if self.npt==np else gm
        gamma = torch.zeros(self.ntot,dtype=torch.float64)
        for i in range(maxiter):
            if verbose: print("\t%-20d%-15.2e"%(i+1,losses[i]))
            Fpt = torch.autograd.grad(Ft,zt,grad_outputs=torch.ones_like(Ft))[0]
            Fp = Fpt.detach().numpy() if self.npt==np else Fpt.detach()
            def mult_Fp(a):
                dima = a.ndim
                assert dima==1 or dima==2
                if dima==1: a = a[:,None]
                m = a.size(1) 
                b = torch.empty((self.ntot,m),dtype=a.dtype)
                for j in range(self.nr):
                    bsl,bsh = self.bs_cumsum[j],self.bs_cumsum[j+1]
                    nsl,nsh = self.n_cumsum[j],self.n_cumsum[j+1]
                    b[nsl:nsh] = (Fp[bsl:bsh,None]*a[bsl:bsh]).reshape((self.tvec[j],self.ns[j],m)).sum(0)
                return b[:,0] if dima==1 else b 
            def mult_tFp(b):
                dimb = b.ndim
                assert dimb==1 or dimb==2 
                if dimb==1: b = b[:,None]
                m = b.size(1)
                a = torch.empty((self.length,m),dtype=b.dtype)
                for j in range(self.nr):
                    bsl,bsh = self.bs_cumsum[j],self.bs_cumsum[j+1]
                    nsl,nsh = self.n_cumsum[j],self.n_cumsum[j+1]
                    a[bsl:bsh] = (Fp[bsl:bsh].reshape((self.tvec[j],self.ns[j],1))*b[nsl:nsh]).flatten(end_dim=1)
                return a[:,0] if dimb==1 else a
            Fpz = mult_Fp(z)
            diff = y-F+Fpz/(1+relaxation)
            Theta_red = mult_Fp(mult_Fp(Theta.T).T)
            if fast_flag:
                def multiply(x):
                    t1 = mult_tFp(x) 
                    t2 = self@t1 
                    t3 = mult_Fp(t2)
                    t4 = t3
                    return t4
            else:
                def multiply(x):
                    return Theta_red@x
            if solver=="CHOL":
                L_chol = torch.linalg.cholesky(Theta_red)
                gamma = torch.cholesky_solve(diff[:,None],L_chol)[:,0]
            else: # (P)CG
                if solver=="CG":
                    precond_solve = False
                elif solver=="PCG-PPCHOL":
                    ddiag = 1e-8*self.npt.ones_like(Theta_red.diagonal())
                    Lk = ppchol(
                        A = Theta_red-self.npt.diag(ddiag),
                        rank = ppchol_rank,
                        rtol = ppchol_rtol,
                        atol = ppchol_atol,
                        return_pivots=False)
                    k = Lk.size(1)
                    pL = torch.linalg.cholesky(torch.eye(k,dtype=float)+(Lk.T/ddiag)@Lk)
                    def precond_solve(x):
                        return solve_ppchol(x,Lk,pL,ddiag)
                    if verbose>1:
                        pmat = precond_solve(Theta_red)
                        cond_Theta_red = torch.linalg.cond(Theta_red)
                        cond_pmat = torch.linalg.cond(pmat) 
                        print("\t\tLk.shape = %-15s K(A) = %-10.1eK(P) = %-10.1eK(P)/K(A)=%.1e"%(tuple(Lk.shape),cond_Theta_red,cond_pmat,cond_pmat/cond_Theta_red))
                elif solver=="PCG-JACOBI":
                    ddiag = Theta_red.diagonal()
                    def precond_solve(x):
                        dimx = x.ndim
                        assert dimx==1 or dimx==2
                        if dimx==1: x = x[:,None]
                        y = x/ddiag
                        return y[:,0] if dimx==1 else y
                    if verbose>1:
                        pmat = precond_solve(Theta_red)
                        cond_Theta_red = torch.linalg.cond(Theta_red)
                        cond_pmat = torch.linalg.cond(pmat) 
                        print("\t\tK(A) = %-10.1eK(P) = %-10.1eK(P)/K(A)=%.1e"%(cond_Theta_red,cond_pmat,cond_pmat/cond_Theta_red))
                elif solver=="PCG-SSOR":
                    assert np.isscalar(ssor_omega) and 0<ssor_omega<2
                    ddiag_over_omega = Theta_red.diagonal()/ssor_omega
                    L_ssor = Theta_red.tril(-1)
                    L_ssor.diagonal().copy_(ddiag_over_omega)
                    def precond_solve(x):
                        dimx = x.ndim
                        assert dimx==1 or dimx==2
                        if dimx==1: x = x[:,None]
                        t1 = torch.linalg.solve_triangular(L_ssor,x,upper=False)
                        t2 = t1*ddiag_over_omega[:,None]
                        t3 = torch.linalg.solve_triangular(L_ssor.T,t2,upper=True)
                        t4 = (2-ssor_omega)*t3
                        return t4[:,0] if dimx==1 else t4
                    if verbose>1:
                        pmat = precond_solve(Theta_red)
                        cond_Theta_red = torch.linalg.cond(Theta_red)
                        cond_pmat = torch.linalg.cond(pmat) 
                        print("\t\tK(A) = %-10.1eK(P) = %-10.1eK(P)/K(A)=%.1e"%(cond_Theta_red,cond_pmat,cond_pmat/cond_Theta_red))
                elif solver=="PCG-BLOCKED":
                    L_chol_blocks = np.empty(self.nr,dtype=object) 
                    for si in range(self.nr):
                        sl,sh = self.n_cumsum[si],self.n_cumsum[si+1]
                        L_chol_blocks[si] = torch.linalg.cholesky(Theta_red[sl:sh,sl:sh])
                    def precond_solve(x):
                        dimx = x.ndim
                        assert dimx==1 or dimx==2
                        if dimx==1: x = x[:,None]
                        y = torch.empty_like(x)
                        for si in range(self.nr):
                            sl,sh = self.n_cumsum[si],self.n_cumsum[si+1]
                            y[sl:sh] = torch.cholesky_solve(x[sl:sh],L_chol_blocks[si])
                        return y[:,0] if dimx==1 else y
                    if verbose>1:
                        pmat = precond_solve(Theta_red)
                        cond_Theta_red = torch.linalg.cond(Theta_red)
                        cond_pmat = torch.linalg.cond(pmat) 
                        print("\t\tK(A) = %-10.1eK(P) = %-10.1eK(P)/K(A)=%.1e"%(cond_Theta_red,cond_pmat,cond_pmat/cond_Theta_red))
                else:
                    assert False, "solver parsing error"                    
                gamma,rbackward_norms[i],times[i] = pcg(
                    matmul = multiply,
                    b = diff,
                    precond_solve = precond_solve,
                    x0 = gamma,
                    rtol = pcg_rtol,
                    atol = pcg_atol,
                    maxiter = pcg_maxiter,
                    beta_method = pcg_beta_method,
                    ref_sol = None,
                    npt = self.npt,
                    ckwargs = self.ckwargs)
            delta = self@mult_tFp(gamma)-z/(1+relaxation)
            print(torch.linalg.norm(delta))
            z = z+delta
            zt = torch.from_numpy(z).clone().requires_grad_() if self.npt==np else z.clone().requires_grad_()
            Ft = pde_lhs_wrap(zt) 
            F = Ft.detach().numpy() if self.npt==np else Ft.detach()
            zhist[(i+1),:] = z
            losses[i+1] = self._loss(F,y)
            if losses[i+1]<loss_best:
                z_best = z
        rbackward_norms,times = rbackward_norms[:(i+1)],times[:(i+1)]
        max_pcg_iters = max([len(rbackward_norm) for rbackward_norm in rbackward_norms])
        rbackward_norms_mat = self.npt.nan*self.npt.empty((i+1,max_pcg_iters))
        times_mat = self.npt.nan*self.npt.empty((i+1,max_pcg_iters))
        for l in range(i+1):
            rbackward_norms_mat[l,:len(rbackward_norms[l])] = rbackward_norms[l]
            times_mat[l,:len(times[l])] = times[l] 
        return z_best.detach(),losses[:(i+2)],rbackward_norms_mat,times_mat,zhist
