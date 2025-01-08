from ._pde_gram_matrix import _PDEGramMatrix
from ...discrete_distribution import Lattice,DigitalNetB2
from ..kernel import KernelShiftInvar,KernelDigShiftInvar
from ..gram_matrix import FastGramMatrixLattice,FastGramMatrixDigitalNetB2
import numpy as np 

class FastPDEGramMatrix(_PDEGramMatrix):
    """ Fast Gram Matrix for solving PDEs 
    
    >>> import torch
    >>> d = 2
    >>> dd_obj = Lattice(d,seed=7)
    >>> kernel_obj = KernelShiftInvar(d,torchify=True)
    >>> us = torch.tensor([
    ...     [True,True],
    ...     [True,False],
    ...     [False,True]])
    >>> ns = torch.tensor([2**5,2**3,2**3],dtype=int)
    >>> llbetas = [
    ...     [torch.tensor([[0,0]]),torch.tensor([[2,0],[0,2]])],
    ...     [torch.tensor([[0,0]])],
    ...     [torch.tensor([[0,0]])]]
    >>> llcs = [
    ...     [torch.ones(1),torch.ones(2)],
    ...     [torch.ones(1)],
    ...     [torch.ones(1)]]
    >>> gmpde = FastPDEGramMatrix(kernel_obj,dd_obj,ns,us,llbetas,llcs)
    >>> gmpde._mult_check()

    >>> dd_obj = DigitalNetB2(d,t_lms=32,alpha=2,seed=7)
    >>> kernel_obj = KernelDigShiftInvar(d,alpha=4,torchify=False)
    >>> us = np.array([
    ...     [True,True],
    ...     [True,False],
    ...     [False,True]])
    >>> ns = np.array([2**5,2**3,2**3],dtype=int)
    >>> llbetas = [
    ...     [np.array([[0,0]]),np.array([[1,0],[0,1]])],
    ...     [np.array([[0,0]])],
    ...     [np.array([[0,0]])]]
    >>> llcs = [
    ...     [np.ones(1),np.ones(2)],
    ...     [np.ones(1)],
    ...     [np.ones(1)]]
    >>> gmpde = FastPDEGramMatrix(kernel_obj,dd_obj,ns,us,llbetas,llcs)
    >>> gmpde._mult_check()
    """
    def __init__(self, kernel_obj, dd_obj, ns=None, us=None, llbetas=None, llcs=None, noise=1e-8, adaptive_noise=True, half_comp=True):
        """
        Args:
            kernel_obj (KernelShiftInvar or KernelDigShiftInvar): the kernel to use 
            dd_obj (Lattice or DigitalNetB2): the discrete distribution from which to sample points 
            llbetas (list of lists): list of length equal to the number of regions where each sub-list is the derivatives at that region 
            llcs (list of lists): list of length equal to the number of regions where each sub-list are the derivative coefficients at that region 
            noise (float): nugget term
            ns (np.ndarray or torch.Tensor): vector of number of points on each of the regions 
            us (np.ndarray or torch.Tensor): bool matrix where each row is a region specifying the active dimensions
        """
        assert isinstance(dd_obj,Lattice) or isinstance(dd_obj,DigitalNetB2)
        if us is None: us = kernel_obj.npt.ones((1,kernel_obj.d),dtype=bool)
        self.us = us 
        self.nr = len(self.us) 
        assert self.us.shape==(self.nr,kernel_obj.d) and (self.us[0]==True).all() # and (self.us.sum(1)>0).all()
        assert ns is not None, "require ns is not None"
        self.ns = ns 
        if isinstance(self.ns,int): self.ns = self.ns*kernel_obj.npt.ones(self.nr,dtype=int)
        assert self.ns.shape==(self.nr,) and (self.ns[1:]<=self.ns[0]).all()
        super(FastPDEGramMatrix,self).__init__(kernel_obj,llbetas,llcs)
        if isinstance(dd_obj,Lattice):
            gmii = FastGramMatrixLattice(kernel_obj,dd_obj,self.ns[0].item(),self.ns[0].item(),self.us[0],self.us[0],self.llbetas[0],self.llbetas[0],self.llcs[0],self.llcs[0],noise=0.,adaptive_noise=False)
        elif isinstance(dd_obj,DigitalNetB2):
            gmii = FastGramMatrixDigitalNetB2(kernel_obj,dd_obj,self.ns[0].item(),self.ns[0].item(),self.us[0],self.us[0],self.llbetas[0],self.llbetas[0],self.llcs[0],self.llcs[0],noise=0.,adaptive_noise=False)
        else:
            raise Exception("Invalid dd_obj") 
        self.gms = np.empty((self.nr,self.nr),dtype=object)
        self.gms[0,0] = gmii
        gmii__x_x = gmii._x,gmii.x 
        gmiitype = type(gmii)
        for i in range(1,self.nr):
            self.gms[0,i] = gmiitype(gmii.kernel_obj,dd_obj,gmii.n1,self.ns[i].item(),gmii.u1,self.us[i,:],gmii.lbeta1s,self.llbetas[i],gmii.lc1s_og,self.llcs[i],0.,False,gmii__x_x)
            self.gms[i,0] = gmiitype(gmii.kernel_obj,dd_obj,self.ns[i].item(),gmii.n1,self.us[i,:],gmii.u1,self.llbetas[i],gmii.lbeta1s,self.llcs[i],gmii.lc1s_og,0.,False,gmii__x_x)
            for k in range(1,self.nr):
                self.gms[i,k] = gmiitype(gmii.kernel_obj,dd_obj,self.ns[i].item(),self.ns[k].item(),self.us[i,:],self.us[k,:],self.llbetas[i],self.llbetas[k],self.llcs[i],self.llcs[k],0.,False,gmii__x_x)
        bs = [self.gms[i,0].size[0] for i in range(self.nr)]
        self.bs_cumsum = [0]+np.cumsum(bs).tolist() 
        self.length = self.bs_cumsum[-1]
        self.n_cumsum = [0]+np.cumsum(self.ns).tolist()
        self.ntot = self.n_cumsum[-1]
        self.tvec = [self.gms[i,i].t1 for i in range(self.nr)]
        self.cholesky = self.gms[0,0].cholesky
        self.cho_solve = self.gms[0,0].cho_solve
        if adaptive_noise:
            assert (self.llbetas[0][0]==0.).all() and self.llbetas[0][0].shape==(1,self.kernel_obj.d) and (self.llcs[0][0]==1.).all() and self.llcs[0][0].shape==(1,)
            full_traces = [[0. for j in range(self.tvec[i1])] for i1 in range(self.nr)]
            self.trace_ratios = [self.npt.zeros(self.tvec[i1],dtype=float) for i1 in range(self.nr)]
            for i1 in range(self.nr):
                for tt1 in range(self.tvec[i1]):
                    betas_i = self.llbetas[i1][tt1] 
                    for i2 in range(self.nr):
                        for tt2 in range(self.tvec[i2]):
                            betas_j = self.llbetas[i2][tt2]
                            if (betas_i==betas_j).all():
                                cs_i = self.llcs[i1][tt1]
                                cs_j = self.llcs[i2][tt2]
                                assert (cs_i==cs_j).all()
                                nj1 = self.gms[i2,i2].n1
                                full_traces[i1][tt1] += nj1*self.gms[i2,i2].k00diags[tt2]
                    self.trace_ratios[i1][tt1] = full_traces[i1][tt1]/full_traces[0][0]
            for i in range(self.nr):
                for tt in range(self.tvec[i]):
                    self.gms[i,i].lam[tt,tt][0,0,0,:] += noise*self.trace_ratios[i][tt]
        else: 
            assert all(self.gms[i,i].invertible for i in range(self.nr))
            for i in range(self.nr):
                for tt1 in range(self.tvec[i]):
                    self.gms[i,i].lam[tt1,tt1][0,0,0,:] += noise
    def precond_solve(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None] # (l,v)
        assert y.ndim==2 and y.shape[0]==(self.length)
        ysplit = np.split(y,self.bs_cumsum[1:-1]) # (li,v)
        ssplit = [self.gms[i,i].solve(ysplit[i]) for i in range(self.nr)]
        s = self.npt.vstack(ssplit)  # (l,v)
        return s[:,0] if yogndim==1 else s
    def __matmul__(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None] # (l,v)
        assert y.ndim==2 and y.shape[0]==(self.length)
        ysplit = np.split(y,self.bs_cumsum[1:-1]) # (li,v)
        ssplit = [0. for i in range(self.nr)]
        for i in range(self.nr):
            for k in range(self.nr):
                ssplit[i] += self.gms[i,k]@ysplit[k]
        s = self.npt.vstack(ssplit) 
        return s[:,0] if yogndim==1 else s
    def get_full_gram_matrix(self):
        return self@self.npt.eye(self.length,dtype=float)
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
        _xs = [None]*self.nr
        for i in range(self.nr):
            if not hasattr(self.gms[i,0],"_x1"): self.gms[i,0]._set__x1__x2()
            _xs[i] = self.gms[i,0].clone(self.gms[i,0]._x1)
        return _xs
    def get_xs(self):
        return [self.gms[i,0]._convert__x_to_x(_x) for i,_x in enumerate(self._get__xs())]
      