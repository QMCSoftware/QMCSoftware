import numpy as np 
import time 

def pcg(matmul, b, x0=None, rtol=1e-8, atol=0., maxiter=None, precond_solve=False, ref_solver=False, npt=np, ckwargs={}):
        """
        Preconditioned Conjugate Gradient (PCG) method, see https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
        
        Args:
            matmul (function): multiply Ax 
            b (np.ndarray or torch.Tensor): right hand side 
            x0 (np.ndarray or torch.Tensor): initial guess
            rtol, atol (float): require norm(b-A@x)<=max(rtol*norm(b),atol)
            maxiter (int): maximum number of iterations 
            precond_solve (function): function which solves the system Px = b for x where P is the preconditioner. 
            ref_solver (function): function used to compute the reference solution 
            npt (numpy or torch): module for numerical computations 
            ckwargs (dict): keyword arguments to pass when constructing a tensor e.g. ckwargs={"device":"cpu"}
        """
        n = len(b)
        if maxiter is None: maxiter = n 
        assert 0<maxiter<=n
        assert b.shape==(n,)
        bnorm = npt.linalg.norm(b)
        x = npt.zeros(n,dtype=float,**ckwargs) if x0 is None else x0 
        assert x.shape==(n,)
        assert rtol>=0 and atol>=0
        residtol = max(rtol*npt.linalg.norm(b),atol)
        assert precond_solve is False or callable(precond_solve)
        if precond_solve is False: 
            precond_solve = lambda r: r 
        backward_norms = npt.zeros(maxiter+1,dtype=float,**ckwargs)
        if ref_solver is not False:
            assert callable(ref_solver)
            tr0 = time.perf_counter()
            xref = ref_solver(b)
            tr = time.perf_counter()-tr0
            xrefnorm = npt.linalg.norm(xref)
            forward_norms = npt.zeros(maxiter+1,dtype=float,**ckwargs)
            forward_norms[0] = npt.linalg.norm(xref) if x0 is None else npt.linalg.norm(xref-x)
        ref_sol = callable(ref_solver)
        times = npt.zeros(maxiter+1,dtype=float,**ckwargs)
        t0 = time.perf_counter()
        r = b if x0 is None else b-matmul(x)
        z = precond_solve(r)
        rz = r@z
        p = z
        backward_norms[0] = npt.linalg.norm(r)
        for i in range(1,maxiter+1):
            Ap = matmul(p)
            alpha = rz/(p@Ap)
            x = x+alpha*p
            r = r-alpha*Ap
            backward_norms[i] = npt.linalg.norm(r)
            if ref_sol: forward_norms[i] = npt.linalg.norm(x-xref)
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