from ..discrete_distribution import Lattice,DigitalNetB2
from .shift_invar_ops import bernoulli_poly
from .dig_shift_invar_ops import weighted_walsh_funcs
import numpy as np 

class _ProdKernel(object):
    def __init__(self, d, alpha, lengthscales, scale, torchify):
        self.torchify = torchify 
        if self.torchify:
            import torch 
            self.npt = torch 
        else: 
            self.npt = np
        assert isinstance(alpha,int) or alpha.shape==(d,)
        if isinstance(alpha,int):
            alpha = alpha*self.npt.ones(d,dtype=int)
        assert isinstance(lengthscales,float) or lengthscales.shape==(d,)
        assert isinstance(scale,float)
        self.d = d 
        self.alpha = alpha
        if isinstance(lengthscales,float):
            lengthscales = lengthscales*self.npt.ones(d)
        self.lengthscales = lengthscales
        self.scale = scale 
    def eval(self, *args, **kwargs):
        return self.__call__(*args,**kwargs)
    def __call__(self, x1, x2, beta1s=0, beta2s=0, c1s=1., c2s=1.):
        """
        Args:
            x1 (np.ndarray or torch.Tensor): n1 x d array of first inputs to kernel 
            x2 (np.ndarray or torch.Tensor): n2 x d array of second inputs to kernel
            beta1s (np.ndarray or torch.Tensor): m1 x d array of first derivative orders
            beta2s (np.ndarray or torch.Tensor): m2 x d array of first derivative orders
            c1s (np.ndarray or torch.Tensor): length m1 vector of derivative coefficients 
            c2s (np.ndarray or torch.Tensor): length m2 vector of derivative coefficients
        """
        n1,n2 = len(x1),len(x2)
        assert x1.shape==(n1,self.d) and x2.shape==(n2,self.d)
        delta = self.x1_ominus_x2(x1[:,None,:],x2[None,:,:]) # n1 x n2 x d
        assert delta.ndim==3 and delta.shape[2]==self.d
        beta1s,beta2s,c1s,c2s,m1,m2 = self._parse_betas_cs(beta1s,beta2s,c1s,c2s)
        inds,idxs,consts = self.inds_idxs_consts(beta1s[:,None,:],beta2s[None,:,:]) # m1 x m2 x n1 x n2 x d
        k = self.npt.ones((m1,m2,n1,n2,self.d))
        self.eval_low_low(k,delta,inds,idxs,consts,m1,m2) # m1 x m2 x n1 x n2 x d
        kcomb = (inds[:,:,None,None,:]+self.lengthscales*k).prod(-1) # m1 x m2 x n1 x n2
        kcomb = (c1s[:,None,None,None]*c2s[None,:,None,None]*kcomb).sum((0,1)) # n1 x n2
        return self.scale*kcomb
    def eval_low_u_noscale(self, u, delta_u, inds, idxs, consts, m1, m2, n1, n2, d_u):
        k_u = self.npt.ones((m1,m2,n1,n2,d_u))
        self.eval_low_low(k_u,delta_u,inds[:,:,u],idxs[:,:,u],consts[:,:,u],m1,m2)
        kcomb_u = (inds[:,:,None,None,u]+self.lengthscales[u]*k_u).prod(-1) # m1 x m1 x n1 x n2
        return kcomb_u
    def _parse_betas_cs(self, beta1s, beta2s, c1s, c2s):
        if isinstance(beta1s,int): beta1s = beta1s*self.npt.ones(self.d,dtype=int)
        if isinstance(beta2s,int): beta2s = beta2s*self.npt.ones(self.d,dtype=int)
        beta1s = self.npt.atleast_2d(beta1s)
        beta2s = self.npt.atleast_2d(beta2s)
        m1 = len(beta1s) 
        m2 = len(beta2s)
        assert beta1s.shape==(m1,self.d) and beta2s.shape==(m2,self.d)
        if isinstance(c1s,float): c1s = c1s*self.npt.ones(m1)
        if isinstance(c2s,float): c2s = c2s*self.npt.ones(m2)
        assert c1s.shape==(m1,) and c2s.shape==(m2,)
        return beta1s,beta2s,c1s,c2s,m1,m2
    
class KernelShiftInvar(_ProdKernel):
    """
    Shift invariant kernel with product weights and derivatives 

    >>> d = 3
    >>> x = Lattice(d,randomize="shift",order="linear",seed=7).gen_samples(8)
    >>> k = KernelShiftInvar(d)
    >>> with np.printoptions(precision=2):
    ...     k(x,x)
    array([[27.05,  0.41,  0.99, -2.41, -0.99, -2.41,  0.99,  0.41],
           [ 0.41, 27.05,  0.41,  0.99, -2.41, -0.99, -2.41,  0.99],
           [ 0.99,  0.41, 27.05,  0.41,  0.99, -2.41, -0.99, -2.41],
           [-2.41,  0.99,  0.41, 27.05,  0.41,  0.99, -2.41, -0.99],
           [-0.99, -2.41,  0.99,  0.41, 27.05,  0.41,  0.99, -2.41],
           [-2.41, -0.99, -2.41,  0.99,  0.41, 27.05,  0.41,  0.99],
           [ 0.99, -2.41, -0.99, -2.41,  0.99,  0.41, 27.05,  0.41],
           [ 0.41,  0.99, -2.41, -0.99, -2.41,  0.99,  0.41, 27.05]])
    >>> import torch 
    >>> xt = torch.from_numpy(x).float()
    >>> kt = KernelShiftInvar(d,torchify=True)
    >>> kt(xt,xt[[0]])
    tensor([[27.0537],
            [ 0.4142],
            [ 0.9942],
            [-2.4140],
            [-0.9942],
            [-2.4140],
            [ 0.9942],
            [ 0.4142]])
    """
    def __init__(self, dimension, alpha=5, lengthscales=1., scale=1., torchify=False):
        """
        Args:
            dimension (int): dimension of the input
            alpha (np.ndarray or torch.Tensor): vector of smoothness parameters of length d 
            lengthscales (np.ndarray or torch.Tensor): vector of length scales of length d 
            scale (float): global scale 
            torchify (bool): wheather or not to use PyTorch backend
        """
        super(KernelShiftInvar,self).__init__(dimension,alpha,lengthscales,scale,torchify)
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
    
class KernelDigShiftInvar(_ProdKernel):
    """
    Digitally Shift invariant kernel with product weights and derivatives 
    
    >>> d = 3
    >>> dnb2 = DigitalNetB2(d,randomize="LMS_DS",seed=7)
    >>> xb = dnb2.gen_samples(8,return_binary=True)
    >>> k = KernelDigShiftInvar(d,dnb2.t_lms)
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
    >>> kt = KernelDigShiftInvar(d,dnb2.t_lms,torchify=True)
    >>> kt(xb,xb[[0]])
    tensor([[13.6329],
            [ 0.1689],
            [ 0.2086],
            [ 0.8560],
            [ 1.1687],
            [ 0.2796],
            [ 0.4520],
            [ 0.4332]])
    """
    def __init__(self, dimension, t=None, alpha=3, lengthscales=1., scale=1., torchify=False):
        """
        Args:
            dimension (int): dimension of the input
            t (int): number of bits in each integer
            alpha (np.ndarray or torch.Tensor): vector of smoothness parameters of length d 
            lengthscales (np.ndarray or torch.Tensor): vector of length scales of length d 
            scale (float): global scale 
            torchify (bool): wheather or not to use PyTorch backend
        """
        super(KernelDigShiftInvar,self).__init__(dimension,alpha,lengthscales,scale,torchify) 
        self.t = t
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