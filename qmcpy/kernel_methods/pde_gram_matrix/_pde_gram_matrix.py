import numpy as np 

class _PDEGramMatrix(object):                 
    def _mult_check(self):
        rng = np.random.Generator(np.random.SFC64(7))
        y = rng.uniform(size=(self.length,2))
        gmatfull = self.get_full_gram_matrix()
        assert np.allclose(self@y[:,0],gmatfull@y[:,0],atol=1e-12)
        assert np.allclose(self@y,gmatfull@y,atol=1e-12)
    def multiply(self, *args, **kwargs):
        return self.__matmul__(*args, **kwargs)
    def pcg(self, b, x0=None, rtol=1e-5, atol=0., maxiter=None, precond=True):
        """
        Preconditioned Conjugate Gradient (PCG) method, see https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
        
        b (np.ndarray or torch.Tensor): right hand side 
        x0 (np.ndarray or torch.Tensor): initial guess
        rtol, atol (float): require norm(b-A@x)<=max(rtol*norm(b),atol)
        maxiter (int): maximum number of iterations 
        precond (bool or str): flag to use preconditioner. 
            If true, the preconditioner is A.precond_solve. 
            If a str s is passed in, the preconditioned system solver is A.s
        """
        n = len(b)
        assert b.shape==(n,)
        x = self.npt.zeros_like(b) if x0 is None else x0 
        assert x.shape==(n,)
        assert rtol>=0 and atol>=0
        residtol = max(rtol*self.npt.linalg.norm(b),atol)
        if maxiter is None: maxiter = n 
        assert 0<maxiter<=n
        assert precond is True or precond is False or isinstance(precond,str) 
        if precond is False: 
            precond_solve = lambda r: r 
        elif precond is True:
            precond_solve = self.precond_solve
        elif isinstance(precond,str):
            precond_solve = getattr(self,precond)
        rnorms = self.npt.zeros(maxiter+1)
        r = b if x0 is None else b-self@x
        z = precond_solve(r)
        rz = r@z
        p = z
        rnorms[0] = self.npt.linalg.norm(r)
        for i in range(1,maxiter+1):
            Ap = self@p
            alpha = rz/(p@Ap)
            x = x+alpha*p
            r = r-alpha*Ap
            rnorms[i] = self.npt.linalg.norm(r)
            if rnorms[i]<=residtol: break
            z = precond_solve(r)
            rznew = r@z
            beta = rznew/rz 
            rz = rznew
            p = z+beta*p
        return x,rnorms[:(i+1)]
    def _solve(self, y):
        if not hasattr(self,"l_chol"): 
            self._init_invertibile()
        return self.cho_solve(self.l_chol,y)
    def condition_number(self):
        return self.npt.linalg.cond(self.gm)