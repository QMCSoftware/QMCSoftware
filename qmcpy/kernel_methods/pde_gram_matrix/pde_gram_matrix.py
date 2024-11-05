from ._pde_gram_matrix import _PDEGramMatrix
from ...discrete_distribution import IIDStdUniform
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
    def __init__(self, xs, kernel_obj, llbetas, llcs, noise=1e-8):
        """
        Args:
            xs (list of numpy.ndarray or torch.Tensor): locations at which the regions are sampled 
            kernel_obj (_Kernel): the kernel to use
            llbetas (list of lists): list of length equal to the number of regions where each sub-list is the derivatives at that region 
            llcs (list of lists): list of length equal to the number of regions where each sub-list are the derivative coefficients at that region 
            noise (float): nugget term
        """
        assert isinstance(xs,list)
        self.nr = len(xs)
        self.npt = kernel_obj.npt
        assert isinstance(llbetas,list) and all(isinstance(lbetas,list) for lbetas in llbetas) and len(llbetas)==self.nr
        assert isinstance(llcs,list) and all(isinstance(lcs,list) for lcs in llcs) and len(llcs)==self.nr
        gms = np.empty((self.nr,self.nr),dtype=object)
        for i,k in itertools.product(range(self.nr),range(self.nr)):
            gms[i,k] = GramMatrix(xs[i],xs[k],kernel_obj,llbetas[i],llbetas[k],llcs[i],llcs[k],noise)
        self.length = np.sum([gms[i,0].size[0] for i in range(self.nr)])
        self.gm = self.npt.vstack([self.npt.hstack([gms[i,k].gm for k in range(self.nr)]) for i in range(self.nr)])
    def _set_diag(self):
        self.gm_diag = self.gm.diagonal()[:,None]
    def precond_solve(self, y):
        if not hasattr(self,"gm_diag"): self._set_diag()
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None] # (l,v)
        assert y.ndim==2 and y.shape[0]==(self.length)
        s = s/self.gm_diag
        return s[:,0] if yogndim==1 else s
    def __matmul__(self, y):
        return self.gm@y
    def get_full_gram_matrix(self):
        return self.gm.copy()
    
