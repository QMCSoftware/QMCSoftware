from ..pcg_module import pcg,PPCholPrecond
import torch.linalg
import scipy.linalg
import numpy as np 
import time 

class _PDEGramMatrix(object):
    def __init__(self, kernel_obj, llbetas, llcs):
        self.kernel_obj = kernel_obj
        self.d = self.kernel_obj.d
        self.npt = self.kernel_obj.npt
        self.ckwargs = self.kernel_obj.ckwargs
        self.llbetas,self.llcs,_inferred_nr = self._parse_llbetas_llcs(llbetas,llcs)
        assert _inferred_nr==self.nr
        self.fast_flag = type(self).__name__=="FastPDEGramMatrix"
        self._l_chol = None 
    @property
    def l_chol(self):
        if self._l_chol is None: 
            self._l_chol = self.cholesky(self.full_mat)
        return self._l_chol
    def _parse_llbetas_llcs(self, llbetas, llcs):
        if llbetas is None: llbetas = [[self.npt.zeros((1,self.d),dtype=int)]]
        if llcs is None: llcs = [[self.npt.ones(len(betas)) for betas in lbetas] for lbetas in llbetas]
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
        if self.npt!=np: 
            import torch
            y = torch.from_numpy(y)
        assert np.allclose(self@y[:,0],self.full_mat@y[:,0],atol=1e-12)
        assert np.allclose(self@y,self.full_mat@y,atol=1e-12)
    def multiply(self, *args, **kwargs):
        return self.__matmul__(*args, **kwargs)
    def _solve(self, y):
        return self.cho_solve(self.l_chol,y)
    def _loss(self, F, y):
        diff = y-F
        loss = self.npt.sqrt((diff**2).mean())
        return loss
    def pde_opt_gauss_newton(self, pde_lhs, pde_rhs, maxiter=8, relaxation=0., verbose=1, precond_setter=None, pcg_kwargs={}, 
                             store_L_chol_hist=False, custom_lin_solver=None, z0=None, store_pcg_data=False):
        t0 = time.perf_counter()
        use_pcg = precond_setter is not None
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
        losses = self.npt.nan*self.npt.ones(maxiter+1)
        times = torch.zeros((maxiter+1),dtype=torch.float64)
        y = pde_rhs_wrap(self.xs) 
        zhist = torch.zeros((maxiter+1,self.length),dtype=torch.float64)
        zt = torch.zeros(self.length,dtype=torch.float64) if z0 is None else z0
        zt.requires_grad_()
        Ft = pde_lhs_wrap(zt)
        F = Ft.detach().numpy() if self.npt==np else Ft.detach()
        self.z = zt.detach().numpy() if self.npt==np else zt.detach()
        zhist[0,:] = self.z
        losses[0] = self._loss(F,y)
        loss_best = losses[0] 
        z_best = self.z
        gamma = torch.zeros(self.ntot,dtype=torch.float64)
        if store_L_chol_hist:
            L_chol_hist = torch.nan*torch.empty((maxiter,len(y),len(y)),dtype=torch.float64)
        if store_pcg_data:
            pcg_data = [None]*maxiter
        times[0] = time.perf_counter()-t0
        for i in range(maxiter):
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
            Fpz = mult_Fp(self.z)
            diff = y-F+Fpz
            self.rtheta = ReducedTheta(self,mult_Fp,mult_tFp,relaxation)
            if custom_lin_solver is not None:
                gamma = custom_lin_solver(self.z,diff)
            elif not use_pcg: # Cholesky factorization
                L_chol = torch.linalg.cholesky(self.rtheta.full_mat)
                gamma = torch.cholesky_solve(diff[:,None],L_chol)[:,0]
                if store_L_chol_hist:
                    L_chol_hist[i] = L_chol
            else: # (P)CG
                precond = precond_setter(self)
                gamma,pcg_data_i = pcg(self.rtheta,diff,precond,x0=gamma,ref_sol=None,ckwargs=self.ckwargs,**pcg_kwargs)
                if store_pcg_data:
                    pcg_data[i] = pcg_data_i
            delta = self@mult_tFp(gamma)-self.z
            self.z = self.z+delta
            zt = torch.from_numpy(self.z).clone().requires_grad_() if self.npt==np else self.z.clone().requires_grad_()
            Ft = pde_lhs_wrap(zt) 
            F = Ft.detach().numpy() if self.npt==np else Ft.detach()
            zhist[(i+1),:] = self.z
            losses[i+1] = self._loss(F,y)
            if losses[i+1]<loss_best:
                z_best = self.z
            times[i+1] = time.perf_counter()-t0
            if verbose:
                # header an initial loss 
                if i==0:
                    header = "    %-15s%-15s%-15s"%("iter (%d max)"%maxiter,"loss","time")
                    if use_pcg:
                        if verbose>1: header += precond._log_header()
                        header += "%-15s%s"%("PCG rberror","PCG steps (%d max)"%len(gamma))
                    log = "    %-15d%-15.2e%-15.2e"%(0,losses[0],times[0])
                    print(header)
                    print(log)
                # loss for this iteration
                log = "    %-15d%-15.2e%-15.2e"%(i+1,losses[i+1],times[i+1])
                if use_pcg:
                    if verbose>1: log += precond._log(self.rtheta.full_mat) # costly condition number computations
                    log += "%-15.1e%d"%(pcg_data_i["rbackward_norms"][-1],len(pcg_data_i["rbackward_norms"])-1)
                print(log)
        z_best = z_best.detach()
        delattr(self,"rtheta")
        delattr(self,"z")
        data = {
            "z_best": z_best,
            "losses": losses[:(i+2)], 
            "times": times, 
            "zhist": zhist,
            "solver": "CHOL" if not use_pcg else "PCG-%s"%type(precond).__name__}
        if store_L_chol_hist:
            data["L_chol_hist"] = L_chol_hist
        if store_pcg_data:
            data["pcg_data"] = pcg_data
        return z_best,data

class ReducedTheta():
    def __init__(self, pde_gm, mult_Fp, mult_tFp, relaxation):
        import torch
        self.pde_gm = pde_gm
        self.mult_Fp = mult_Fp
        self.mult_tFp = mult_tFp
        self.npt = torch
        self.relaxation = relaxation
        self._full_mat = None
    @property
    def full_mat(self):
        if self._full_mat is None:
            fm_minus_relaxation= self.mult_Fp(self.mult_Fp(self.pde_gm.full_mat.T).T)
            self._full_mat = fm_minus_relaxation+self.relaxation*self.npt.eye(len(fm_minus_relaxation),dtype=fm_minus_relaxation.dtype)
        return self._full_mat
    def __matmul__(self, x):
        return self.mult_Fp(self.pde_gm@self.mult_tFp(x))+self.relaxation*x if self.pde_gm.fast_flag else self.full_mat@x