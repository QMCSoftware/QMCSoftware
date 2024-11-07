from ._pde_gram_matrix import _PDEGramMatrix
from ...discrete_distribution import Lattice,DigitalNetB2
from ..kernel import KernelShiftInvar
from ..gram_matrix import FastGramMatrixLattice,FastGramMatrixDigitalNetB2
import numpy as np 

class FastPDEGramMatrix(_PDEGramMatrix):
    """ Fast Gram Matrix for solving PDEs 
    
    >>> d = 2
    >>> lat_obj = Lattice(d,seed=7)
    >>> kernel_si = KernelShiftInvar(d)
    >>> us = np.array([
    ...     [True,True],
    ...     [True,False],
    ...     [False,True]])
    >>> ns = np.array([2**5,2**3,2**3],dtype=int)
    >>> llbetas = [
    ...     [np.array([[1,0],[0,1]]),np.array([[0,0]])],
    ...     [np.array([[0,0]])],
    ...     [np.array([[0,0]])]]
    >>> llcs = [
    ...     [np.ones(2),np.ones(1)],
    ...     [np.ones(1)],
    ...     [np.ones(1)]]
    >>> gmpde = FastPDEGramMatrix(lat_obj,kernel_si,ns=ns,us=us,llbetas=llbetas,llcs=llcs)
    >>> gmpde._mult_check()
    """
    def __init__(self, dd_obj, kernel_obj, ns, us, llbetas, llcs, noise=1e-8):
        """
        Args:
            dd_obj (Lattice or DigitalNetB2): the discrete distribution from which to sample points 
            kernel_obj (KernelShiftInvar or KernelDigShiftInvar): the kernel to use 
            n_bs (np.ndarray or torch.Tensor): vector of number of points on each of the regions 
            us (np.ndarray or torch.Tensor): bool matrix where each row is a region specifying the active dimensions
            llbetas (list of lists): list of length equal to the number of regions where each sub-list is the derivatives at that region 
            llcs (list of lists): list of length equal to the number of regions where each sub-list are the derivative coefficients at that region 
            noise (float): nugget term
        """
        self.npt = kernel_obj.npt
        self.ckwargs = kernel_obj.ckwargs
        assert isinstance(dd_obj,Lattice) or isinstance(dd_obj,DigitalNetB2)
        self.us = self.npt.atleast_2d(us) 
        self.nr = len(self.us) 
        assert self.us.shape==(self.nr,kernel_obj.d) and (self.us.sum(1)>0).all() and (self.us[0]==True).all()
        self.ns = ns 
        if isinstance(self.ns,int): self.ns = self.ns*self.npt.ones(self.nr,dtype=int)
        assert self.ns.shape==(self.nr,) and (self.ns[1:]<=self.ns[0]).all()
        assert isinstance(llbetas,list) and all(isinstance(lbetas,list) for lbetas in llbetas) and len(llbetas)==self.nr
        assert isinstance(llcs,list) and all(isinstance(lcs,list) for lcs in llcs) and len(llcs)==self.nr
        if isinstance(dd_obj,Lattice):
            gmii = FastGramMatrixLattice(dd_obj,kernel_obj,self.ns[0].item(),self.ns[0].item(),self.us[0],self.us[0],llbetas[0],llbetas[0],llcs[0],llcs[0],noise)
        elif isinstance(dd_obj,DigitalNetB2):
            gmii = FastGramMatrixDigitalNetB2(dd_obj,kernel_obj,self.ns[0].item(),self.ns[0].item(),self.us[0],self.us[0],llbetas[0],llbetas[0],llcs[0],llcs[0],noise)
        else:
            raise Exception("Invalid dd_obj") 
        self.gms = np.empty((self.nr,self.nr),dtype=object)
        self.gms[0,0] = gmii
        gmii__x_x = gmii._x,gmii.x 
        gmiitype = type(gmii)
        for i in range(1,self.nr):
            self.gms[0,i] = gmiitype(dd_obj,gmii.kernel_obj,gmii.n1,self.ns[i].item(),gmii.u1,self.us[i,:],gmii.lbeta1s,llbetas[i],gmii.lc1s_og,llcs[i],gmii.noise,gmii__x_x)
            self.gms[i,0] = gmiitype(dd_obj,gmii.kernel_obj,self.ns[i].item(),gmii.n1,self.us[i,:],gmii.u1,llbetas[i],gmii.lbeta1s,llcs[i],gmii.lc1s_og,gmii.noise,gmii__x_x)
            for k in range(1,self.nr):
                self.gms[i,k] = gmiitype(dd_obj,gmii.kernel_obj,self.ns[i].item(),self.ns[k].item(),self.us[i,:],self.us[k,:],llbetas[i],llbetas[k],llcs[i],llcs[k],gmii.noise,gmii__x_x)
        bs = [self.gms[i,0].size[0] for i in range(self.nr)]
        self.bs_cumsum = np.cumsum(bs).tolist() 
        self.length = self.bs_cumsum[-1]
        self.bs_cumsum = self.bs_cumsum[:-1]
        self.cholesky = self.gms[0,0].cholesky
        self.cho_solve = self.gms[0,0].cho_solve
    def precond_solve(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None] # (l,v)
        assert y.ndim==2 and y.shape[0]==(self.length)
        ysplit = np.split(y,self.bs_cumsum) # (li,v)
        ssplit = [self.gms[i,i].solve(ysplit[i]) for i in range(self.nr)]
        s = self.npt.vstack(ssplit)  # (l,v)
        return s[:,0] if yogndim==1 else s
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
    def _init_invertibile(self):
        gm = self.get_full_gram_matrix()
        self.l_chol = self.cholesky(gm)
    def condition_number(self):
        gm = self.get_full_gram_matrix()
        return self.npt.linalg.cond(gm)
    def precond_condition_number(self):
        gm = self.get_full_gram_matrix()
        pgm = self.precond_solve(gm)
        return self.npt.linalg.cond(pgm)
    def _get__xs(self):
        _x = self.gms[0,0]._x
        _xs = [self.gms[0,0].clone(_x)[:self.ns[i]] for i in range(self.nr)]
        for i in range(self.nr):
            _xs[i][:,~self.us[i]] = 0.
        return _xs
      