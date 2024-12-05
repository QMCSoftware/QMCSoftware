from ._pde_gram_matrix import _PDEGramMatrix
from ..util import ppchol,solve_ppchol
from ...discrete_distribution import IIDStdUniform
from ...discrete_distribution._discrete_distribution import DiscreteDistribution
from ..kernel import KernelGaussian
from ..gram_matrix import GramMatrix
import numpy as np 
import itertools

class PDEGramMatrix(_PDEGramMatrix):
    """ Fast Gram Matrix for solving PDEs 
    
    >>> d = 2
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> nint = 5
    >>> nb = 3
    >>> xint = rng.uniform(size=(nint,d))
    >>> xb = np.empty((4*nb,d))
    >>> # bottom
    >>> xb[:nb,0] = rng.uniform(size=nb)
    >>> xb[:nb,1] = 0.
    >>> # top 
    >>> xb[nb:(2*nb),0] = rng.uniform(size=nb)
    >>> xb[nb:(2*nb),1] = 1.
    >>> # left
    >>> xb[(2*nb):(3*nb),1] = rng.uniform(size=nb)
    >>> xb[(2*nb):(3*nb),0] = 0.
    >>> # right
    >>> xb[(3*nb):,1] = rng.uniform(size=nb)
    >>> xb[(3*nb):,0] = 1.
    >>> kernel_gaussian = KernelGaussian(d)
    >>> llbetas = [
    ...     [np.array([[1,0],[0,1]]),np.array([[0,0]])],
    ...     [np.array([[0,0]])]]
    >>> llcs = [
    ...     [np.ones(2),np.ones(1)],
    ...     [np.ones(1)]]
    >>> gmpde = PDEGramMatrix([xint,xb],kernel_gaussian,llbetas=llbetas,llcs=llcs)
    >>> gmpde._mult_check()
    """
    def __init__(self, xs, kernel_obj, llbetas, llcs, noise=1e-8, ns=None, us=None, adaptive_noise=True, half_comp=True):
        """
        Args:
            xs (DiscreteDisribution or list of numpy.ndarray or torch.Tensor): locations at which the regions are sampled 
            kernel_obj (_Kernel): the kernel to use
            llbetas (list of lists): list of length equal to the number of regions where each sub-list is the derivatives at that region 
            llcs (list of lists): list of length equal to the number of regions where each sub-list are the derivative coefficients at that region 
            noise (float): nugget term
            ns (np.ndarray or torch.Tensor): vector of number of points on each of the regions. 
                Only used when a DiscreteDistribution is passed in for xs
            us (np.ndarray or torch.Tensor): bool matrix where each row is a region specifying the active dimensions
                Only used when a DiscreteDistribution is passed in for xs
            half_comp (bool or numpy.ndarray or torch.Tensor): if True, put project half the points to the 0 boundary and half the points to the 1 boundary
        """
        if isinstance(xs,DiscreteDistribution):
            dd_obj = xs
            assert ns is not None and ns.ndim==1 and (ns[1:]<=ns[0]).all()
            nr = len(ns)
            assert us is not None and us.ndim==2 and us.shape==(nr,kernel_obj.d) and (us[0]==True).all() # and (us.sum(1)>0).all()
            x = dd_obj.gen_samples(ns[0])
            if kernel_obj.torchify: 
                import torch 
                x = torch.from_numpy(x)
                xs = [x[:n].clone() for n in ns]
            else: # numpyify 
                xs = [x[:n].copy() for n in ns]
            if isinstance(half_comp,bool): half_comp = half_comp*torch.ones(len(us),dtype=bool)
            assert len(half_comp)==len(us)
            for i,u in enumerate(us):
                if half_comp[i]:
                    nhalf = ns[i]//2
                    xs[i][:nhalf,~u] = 1. 
                    xs[i][nhalf:,~u] = 0.
                else:
                    xs[i][:,~u] = 0.
        assert isinstance(xs,list)
        self.xs = xs 
        self.nr = len(self.xs)
        self.ns = np.array([len(x) for x in self.xs],dtype=int)
        super(PDEGramMatrix,self).__init__(kernel_obj,llbetas,llcs)
        self.gms = np.empty((self.nr,self.nr),dtype=object)
        for i1,i2 in itertools.product(range(self.nr),range(self.nr)):
            self.gms[i1,i2] = GramMatrix(self.xs[i1],self.xs[i2],self.kernel_obj,self.llbetas[i1],self.llbetas[i2],self.llcs[i1],self.llcs[i2],noise=0.,adaptive_noise=False)
        bs = [self.gms[i,0].size[0] for i in range(self.nr)]
        self.bs_cumsum = [0]+np.cumsum(bs).tolist() 
        self.length = self.bs_cumsum[-1]
        self.n_cumsum = [0]+np.cumsum(self.ns).tolist()
        self.ntot = self.n_cumsum[-1]
        if adaptive_noise:
            assert (llbetas[0][0]==0.).all() and llbetas[0][0].shape==(1,self.kernel_obj.d) and (llcs[0][0]==1.).all() and llcs[0][0].shape==(1,)
            full_traces = [[0. for j in range(self.gms[i1,i1].t1)] for i1 in range(self.nr)]
            self.trace_ratios = [self.npt.zeros(self.gms[i1,i1].t1) for i1 in range(self.nr)]
            for i1 in range(self.nr):
                ni1 = self.gms[i1,i1].n1
                for tt1 in range(self.gms[i1,i1].t1):
                    betas_i = llbetas[i1][tt1] 
                    for i2 in range(self.nr):
                        for tt2 in range(self.gms[i2,i2].t1):
                            betas_j = llbetas[i2][tt2]
                            if (betas_i==betas_j).all():
                                cs_i = llcs[i1][tt1]
                                cs_j = llcs[i2][tt2]
                                assert (cs_i==cs_j).all()
                                nj1 = self.gms[i2,i2].n1
                                full_traces[i1][tt1] += nj1*self.gms[i2,i2].gm[tt2*nj1,tt2*nj1]
                    self.trace_ratios[i1][tt1] = full_traces[i1][tt1]/full_traces[0][0]
            for i in range(self.nr):
                ni1 = self.gms[i,i].n1
                self.gms[i,i].gm += noise*self.npt.diag((self.npt.ones((ni1,self.gms[i,i].t1))*self.trace_ratios[i]).T.flatten())
                self.gm = self.npt.vstack([self.npt.hstack([self.gms[i,k].gm for k in range(self.nr)]) for i in range(self.nr)])
        else:
            assert all(self.gms[i,i].invertible for i in range(self.nr))
            self.gm = self.npt.vstack([self.npt.hstack([self.gms[i,k].gm for k in range(self.nr)]) for i in range(self.nr)])
            self.gm += noise*self.npt.eye(self.length,dtype=float,**self.ckwargs)
        self.cholesky = self.gms[0,0].cholesky
        self.cho_solve = self.gms[0,0].cho_solve
    def _init_precond_solve(self):
        if not hasattr(self,"ddiag"):
            self.ddiag = 1e-8*self.npt.ones_like(self.gm.diagonal())
            self.Lk = ppchol(self.gm-self.npt.diag(self.ddiag),return_pivots=False)
            k = self.Lk.size(1)
            self.pL = self.npt.linalg.cholesky(self.npt.eye(k,dtype=float)+(self.Lk.T/self.ddiag)@self.Lk)
    def precond_solve(self, y):
        return solve_ppchol(y,self.Lk,self.pL,self.ddiag)
    def _set_diag(self):
        self.gm_diag = self.gm.diagonal()[:,None]
    def __matmul__(self, y):
        return self.gm@y
    def get_full_gram_matrix(self):
        return self.gm
    def _init_invertibile(self):
        self.l_chol = self.cholesky(self.gm)
    def condition_number(self):
        return self.npt.linalg.cond(self.gm)
    def get_xs(self):
        return [self.gms[0,0].clone(x) for x in self.xs]
    
    
