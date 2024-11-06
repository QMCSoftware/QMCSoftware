import numpy as np 
import time 

class _PDEGramMatrix(object):                 
    def _mult_check(self):
        rng = np.random.Generator(np.random.SFC64(7))
        y = rng.uniform(size=(self.length,2))
        gmatfull = self.get_full_gram_matrix()
        assert np.allclose(self@y[:,0],gmatfull@y[:,0],atol=1e-12)
        assert np.allclose(self@y,gmatfull@y,atol=1e-12)
    def multiply(self, *args, **kwargs):
        return self.__matmul__(*args, **kwargs)
    def pcg(self, b, x0=None, rtol=1e-5, atol=0., maxiter=None, precond=True, ref_sol=False):
        """
        Preconditioned Conjugate Gradient (PCG) method, see https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
        
        Args:
            b (np.ndarray or torch.Tensor): right hand side 
            x0 (np.ndarray or torch.Tensor): initial guess
            rtol, atol (float): require norm(b-A@x)<=max(rtol*norm(b),atol)
            maxiter (int): maximum number of iterations 
            precond (bool or str): flag to use preconditioner. 
                If true, the preconditioner is A.precond_solve. 
                If a str s is passed in, the preconditioned system solver is A.s
            ref_sol (bool): if True, compute a reference solution and track the relative error (forward error)
        """
        n = len(b)
        if maxiter is None: maxiter = n 
        assert 0<maxiter<=n
        assert b.shape==(n,)
        bnorm = self.npt.linalg.norm(b)
        x = self.npt.zeros(n,dtype=float,**self.ckwargs) if x0 is None else x0 
        assert x.shape==(n,)
        assert rtol>=0 and atol>=0
        residtol = max(rtol*self.npt.linalg.norm(b),atol)
        assert precond is True or precond is False or isinstance(precond,str) 
        if precond is False: 
            precond_solve = lambda r: r 
        elif precond is True:
            precond_solve = self.precond_solve
        elif isinstance(precond,str):
            precond_solve = getattr(self,precond)
        backward_norms = self.npt.zeros(maxiter+1,dtype=float,**self.ckwargs)
        if ref_sol:
            tr0 = time.perf_counter()
            xref = self._solve(b)
            tr = time.perf_counter()-tr0
            xrefnorm = self.npt.linalg.norm(xref)
            forward_norms = self.npt.zeros(maxiter+1,dtype=float,**self.ckwargs)
            forward_norms[0] = self.npt.linalg.norm(xref) if x0 is None else self.npt.linalg.norm(xref-x)
        times = self.npt.zeros(maxiter+1,dtype=float,**self.ckwargs)
        t0 = time.perf_counter()
        r = b if x0 is None else b-self@x
        z = precond_solve(r)
        rz = r@z
        p = z
        backward_norms[0] = self.npt.linalg.norm(r)
        for i in range(1,maxiter+1):
            Ap = self@p
            alpha = rz/(p@Ap)
            x = x+alpha*p
            r = r-alpha*Ap
            backward_norms[i] = self.npt.linalg.norm(r)
            if ref_sol: forward_norms[i] = self.npt.linalg.norm(x-xref)
            times[i] = time.perf_counter()-t0
            if backward_norms[i]<=residtol or i==maxiter: break
            z = precond_solve(r)
            rznew = r@z
            beta = rznew/rz 
            rz = rznew
            p = z+beta*p
        times = times[:(i+1)]
        rbackward_norms = backward_norms[:(i+1)]/bnorm
        if not ref_sol:
            return x,rbackward_norms,times
        else:
            rforward_norms = forward_norms[:(i+1)]/xrefnorm
            return x,rbackward_norms,times,xref,rforward_norms,tr
    def _solve(self, y):
        if not hasattr(self,"l_chol"): 
            self._init_invertibile()
        return self.cho_solve(self.l_chol,y)