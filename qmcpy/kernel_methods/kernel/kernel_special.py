from ._kernel_prod import _KernelProd
from ..util import bernoulli_poly,weighted_walsh_funcs
from ...discrete_distribution import Lattice,DigitalNetB2
import numpy as np

class _KernelProdSpecial(_KernelProd):
    def parsed_call(self, x1, x2, n1, n2, beta1s, beta2s, m1, m2, c1s, c2s, diag_only):
        if diag_only:
            delta = self.x1_ominus_x2(x1[:,None,:],x2[:,None,:])
            assert delta.ndim==3 and delta.shape[2]==self.d
            inds,idxs,consts = self.inds_idxs_consts(beta1s[:,None,:],beta2s[None,:,:])
            k = self.npt.ones((m1,1,n1,1,self.d),dtype=float,**self.ckwargs)
            self.eval_low_low(k,delta,inds,idxs,consts,m1,1)
        else:
            delta = self.x1_ominus_x2(x1[:,None,:],x2[None,:,:]) # n1 x n2 x d
            assert delta.ndim==3 and delta.shape[2]==self.d
            inds,idxs,consts = self.inds_idxs_consts(beta1s[:,None,:],beta2s[None,:,:]) # m1 x m2 x n1 x n2 x d
            k = self.npt.ones((m1,m2,n1,n2,self.d),dtype=float,**self.ckwargs)
            self.eval_low_low(k,delta,inds,idxs,consts,m1,m2) # m1 x m2 x n1 x n2 x d
        kcomb = (inds[:,:,None,None,:]+self.lengthscales*k).prod(-1) # m1 x m2 x n1 x n2
        kcomb = (c1s[:,None,None,None]*c2s[None,:,None,None]*kcomb).sum((0,1)) # n1 x n2
        kmat = self.scale*kcomb
        return kmat[:,0] if diag_only else kmat
    def eval_low_u_noscale(self, u, delta_u, inds, idxs, consts, m1, m2, n1, n2, d_u):
        k_u = self.npt.ones((m1,m2,n1,n2,d_u),dtype=float,**self.ckwargs)
        self.eval_low_low(k_u,delta_u,inds[:,:,u],idxs[:,:,u],consts[:,:,u],m1,m2)
        kcomb_u = (inds[:,:,None,None,u]+self.lengthscales[u]*k_u).prod(-1) # m1 x m1 x n1 x n2
        return kcomb_u

class KernelShiftInvar(_KernelProdSpecial):
    """
    Shift invariant kernel with product weights and derivatives 

    >>> d = 3
    >>> x = Lattice(d,randomize="shift",order="linear",seed=7).gen_samples(8)
    >>> k = KernelShiftInvar(d,torchify=False)
    >>> with np.printoptions(precision=2):
    ...     k(x,x)
    array([[27.22,  0.41,  0.98, -2.41, -0.98, -2.41,  0.98,  0.41],
           [ 0.41, 27.22,  0.41,  0.98, -2.41, -0.98, -2.41,  0.98],
           [ 0.98,  0.41, 27.22,  0.41,  0.98, -2.41, -0.98, -2.41],
           [-2.41,  0.98,  0.41, 27.22,  0.41,  0.98, -2.41, -0.98],
           [-0.98, -2.41,  0.98,  0.41, 27.22,  0.41,  0.98, -2.41],
           [-2.41, -0.98, -2.41,  0.98,  0.41, 27.22,  0.41,  0.98],
           [ 0.98, -2.41, -0.98, -2.41,  0.98,  0.41, 27.22,  0.41],
           [ 0.41,  0.98, -2.41, -0.98, -2.41,  0.98,  0.41, 27.22]])
    >>> import torch 
    >>> xt = torch.from_numpy(x).float()
    >>> kt = KernelShiftInvar(d,torchify=True)
    >>> kt(xt,xt[[0]])
    tensor([[27.2208],
            [ 0.4138],
            [ 0.9768],
            [-2.4126],
            [-0.9776],
            [-2.4126],
            [ 0.9768],
            [ 0.4138]], dtype=torch.float64)
    """
    DEFAULTALPHA = 4
    def _my_init(self, **kwargs):
        if self.torchify:
            self.lgamma = self.npt.lgamma
        else:
            import scipy.special
            self.lgamma = scipy.special.loggamma
    def inds_idxs_consts(self, beta1, beta2):
        beta1pbeta2 = beta1+beta2
        inds = 1.*(beta1pbeta2==0)
        idxs = 2*self.alpha-beta1pbeta2
        consts = (-1)**(self.alpha+beta2+1)*self.npt.exp(2*self.alpha*np.log(2*np.pi)-self.lgamma(idxs+1))
        return inds,idxs,consts
    def x1_ominus_x2(self, x1, x2):
        delta = (x1-x2)%1
        return delta        
    def eval_low_low(self, k, delta, inds, idxs, consts, m1, m2):
        for l1 in range(m1):
            for l2 in range(m2):
                for j in range(delta.shape[2]):
                    k[l1,l2,:,:,j] = consts[l1,l2,j]*bernoulli_poly(idxs[l1,l2,j].item(),delta[:,:,j])
    
class KernelDigShiftInvar(_KernelProdSpecial):
    """
    Digitally Shift invariant kernel with product weights and derivatives 
    
    >>> d = 3
    >>> dnb2 = DigitalNetB2(d,randomize="LMS_DS",seed=7)
    >>> xb = dnb2.gen_samples(8,return_binary=True)
    >>> k = KernelDigShiftInvar(d,t=dnb2.t_lms,torchify=False)
    >>> with np.printoptions(precision=2):
    ...     k(xb,xb)
    array([[13.63,  0.17,  0.21,  0.86,  1.17,  0.28,  0.45,  0.43],
           [ 0.17, 13.63,  0.86,  0.21,  0.28,  1.17,  0.43,  0.45],
           [ 0.21,  0.86, 13.63,  0.17,  0.45,  0.43,  1.17,  0.28],
           [ 0.86,  0.21,  0.17, 13.63,  0.43,  0.45,  0.28,  1.17],
           [ 1.17,  0.28,  0.45,  0.43, 13.63,  0.17,  0.21,  0.86],
           [ 0.28,  1.17,  0.43,  0.45,  0.17, 13.63,  0.86,  0.21],
           [ 0.45,  0.43,  1.17,  0.28,  0.21,  0.86, 13.63,  0.17],
           [ 0.43,  0.45,  0.28,  1.17,  0.86,  0.21,  0.17, 13.63]])
    >>> import torch
    >>> kt = KernelDigShiftInvar(d,torchify=True,t=dnb2.t_lms)
    >>> kt(xb,xb[[0]])
    tensor([[13.6329],
            [ 0.1689],
            [ 0.2086],
            [ 0.8560],
            [ 1.1687],
            [ 0.2796],
            [ 0.4520],
            [ 0.4332]], dtype=torch.float64)
    """
    DEFAULTALPHA = 3
    def _my_init(self, **kwargs):
        self.t = kwargs["t"] if "t" in kwargs else None
    def set_t(self, t):
        self.t = t
    def inds_idxs_consts(self, beta1, beta2):
        beta1pbeta2 = beta1+beta2
        inds = 1.*(beta1pbeta2==0)
        idxs = self.alpha-beta1pbeta2
        consts = (-2)**beta1pbeta2
        return inds,idxs,consts
    def x1_ominus_x2(self, x1, x2):
        delta = x1^x2
        return delta        
    def eval_low_low(self, k, delta, inds, idxs, consts, m1, m2):
        for l1 in range(m1):
            for l2 in range(m2):
                for j in range(delta.shape[2]):
                    k[l1,l2,:,:,j] = consts[l1,l2,j]*weighted_walsh_funcs(idxs[l1,l2,j].item(),delta[:,:,j],self.t)-inds[l1,l2,j]
                    