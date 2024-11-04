import numpy as np 
from ..discrete_distribution import Lattice
from .kernels import KernelShiftInvar 
from .fast_gram_matrix import _FastGramMatrix,FastGramMatrixLattice

class FastPDEGramMatrix(object):
    """ Fast Gram Matrix for solving PDEs 
    
    >>> d = 2
    >>> lbetas_i = [
    ...     np.array([[1,0],[0,1]]),
    ...     np.array([[0,0]])]
    >>> lcs_i = [
    ...     np.ones(2),
    ...     np.ones(1)]
    >>> n_i = 2**5
    >>> lat_obj = Lattice(d,seed=7)
    >>> kernel_si = KernelShiftInvar(d)
    >>> gmii = FastGramMatrixLattice(lat_obj,kernel_si,n_i,n_i,lbeta1s=lbetas_i,lbeta2s=lbetas_i,lc1s=lcs_i,lc2s=lcs_i)
    >>> us_bs = np.array([
    ...     [True,False],
    ...     [False,True]])
    >>> lbetas_bs = [
    ...     [np.array([[0,0]])],
    ...     [np.array([[0,0]])]]
    >>> lcs_bs = [
    ...     [np.ones(1)],
    ...     [np.ones(1)]]
    >>> n_b = 2**3
    >>> gmpde = FastPDEGramMatrix(gmii,n_b,us_bs,lbetas_bs,lcs_bs)
    >>> gmpde._mult_check()
    """
    def __init__(self, gmii, n_b, us_bs=None, lbetas_bs=0, lcs_bs=1.):
        """
        Args:
            gmii (FastGramMatrixLattice or FastGramMatrixDigitalNetB2): the gram matrix for the interior 
            n_bs (np.ndarray or torch.Tensor): vector of number of points on each of the boundary regions 
            us_bs (np.ndarray or torch.Tensor): bool matrix where each row is a region specifying the active dimensions
            lbetas_bs (list of lists): list of length equal to the number of regions where each sub-list is the derivatives at that region 
            lcs_bs (list of lists): list of length equal to the number of regions where each sub-list are the derivative coefficients at that region 
        """
        assert isinstance(gmii,_FastGramMatrix)
        assert (gmii.u1==True).all() and (gmii.u2==True).all() and (gmii.n1==gmii.n2) and gmii.invertible
        self.npt = gmii.npt 
        us_bs = self.npt.atleast_2d(us_bs) 
        nbr = len(us_bs) 
        assert us_bs.shape==(nbr,gmii.d) and (us_bs.sum(1)>0).all()
        us_bs = us_bs
        if isinstance(n_b,int): n_b = n_b*self.npt.ones(nbr,dtype=int)
        assert n_b.shape==(nbr,) and (n_b<=gmii.n1).all() and (n_b>0).all()
        if isinstance(lbetas_bs,int): lbetas_bs = [[lbetas_bs*self.npt.ones((1,gmii.d),dtype=int)] for i in range(nbr)]
        assert isinstance(lbetas_bs,list) and all(isinstance(lbetas_b,list) for lbetas_b in lbetas_bs) and len(lbetas_bs)==nbr
        if isinstance(lcs_bs,int): lcs_bs = [[self.npt.ones(1)] for i in range(nbr)]
        assert isinstance(lcs_bs,list) and all(isinstance(lcs_b,list) for lcs_b in lcs_bs) and len(lcs_bs)==nbr 
        self.nr = 1+nbr 
        self.gms = np.empty((self.nr,self.nr),dtype=object)
        self.gms[0,0] = gmii
        gmii__x_x = gmii._x,gmii.x 
        gmiitype = type(gmii)
        for i in range(nbr):
            self.gms[0,i+1] = gmiitype(gmii__x_x,gmii.kernel_obj,gmii.n1,n_b[i].item(),gmii.u1,us_bs[i,:],gmii.lbeta1s,lbetas_bs[i],gmii.lc1s_og,lcs_bs[i],gmii.noise)
            self.gms[i+1,0] = gmiitype(gmii__x_x,gmii.kernel_obj,n_b[i].item(),gmii.n1,us_bs[i,:],gmii.u1,lbetas_bs[i],gmii.lbeta1s,lcs_bs[i],gmii.lc1s_og,gmii.noise)
            for k in range(nbr):
                self.gms[i+1,k+1] = gmiitype(gmii__x_x,gmii.kernel_obj,n_b[i],n_b[k],us_bs[i,:],us_bs[k,:],lbetas_bs[i],lbetas_bs[k],lcs_bs[i],lcs_bs[k],gmii.noise)
        bs = [self.gms[i,0].size[0] for i in range(self.nr)]
        self.bs_cumsum = np.cumsum(bs).tolist() 
        self.length = self.bs_cumsum[-1]
        self.bs_cumsum = self.bs_cumsum[:-1]
    def precond_solve(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None] # (l,v)
        assert y.ndim==2 and y.shape[0]==(self.length)
        ysplit = np.split(y,self.bs_cumsum) # (li,v)
        ssplit = [self.gms[i,i].solve(ysplit[i]) for i in range(self.nr)]
        s = self.npt.vstack(ssplit)  # (l,v)
        return s[:,0] if yogndim==1 else s
    def multiply(self, *args, **kwargs):
        return self.__matmul__(*args, **kwargs)
    def __matmul__(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None] # (l,v)
        assert y.ndim==2 and y.shape[0]==(self.length)
        ysplit = np.split(y,self.bs_cumsum) # (li,v)
        ssplit = [0. for i in range(self.nr)]
        for i in range(self.nr):
            for k in range(self.nr):
                ssplit[i] += self.gms[i,k]@ysplit[k]
        s = self.npt.vstack(ssplit) 
        return s[:,0] if yogndim==1 else s
    def get_full_gram_matrix(self):
        return self.npt.vstack([self.npt.hstack([self.gms[i,k].get_full_gram_matrix() for k in range(self.nr)]) for i in range(self.nr)])
    def _mult_check(self):
        rng = np.random.Generator(np.random.SFC64(7))
        y = rng.uniform(size=(self.length,2))
        gmatfull = self.get_full_gram_matrix()
        assert np.allclose(self@y[:,0],gmatfull@y[:,0],atol=1e-12)
        assert np.allclose(self@y,gmatfull@y,atol=1e-12)
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
