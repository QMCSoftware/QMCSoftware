from ..discrete_distribution import Lattice,DigitalNetB2
from .shift_invar_ops import bernoulli_poly
from .dig_shift_invar_ops import weighted_walsh_funcs
import numpy as np 

class _ProdKernel(object):
    def __init__(self, d, alpha, lengthscales, scale, beta1, beta2, torchify):
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
        assert isinstance(beta1,int) or beta1.shape==(d,)
        assert isinstance(beta2,int) or beta2.shape==(d,)
        self.d = d 
        self.alpha = alpha
        self.lengthscales = lengthscales
        self.scale = scale 
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.beta1pbeta2 = self.beta1+self.beta2
        self.ind = (self.beta1pbeta2==0)+self.alpha-self.alpha
    def __call__(self, x1, x2):
        """
        Args:
            x1 (np.ndarray or torch.Tensor): n1 x d array of first inputs to kernel 
            x2 (np.ndarray or torch.Tensor): n2 x d array of second inputs to kernel
        """
        n1,n2 = len(x1),len(x2)
        assert x1.shape==(n1,self.d) and x2.shape==(n2,self.d)
        k = self.npt.empty((n1,n2,self.d))
        self._low_call(x1,x2,k)
        kcomb = (self.ind+self.lengthscales*k).prod(2)
        return self.scale*kcomb
    
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
    def __init__(self, dimension, alpha=5, lengthscales=1., scale=1., beta1=0, beta2=0, torchify=False):
        """
        Args:
            dimension (int): dimension of the input
            alpha (np.ndarray or torch.Tensor): vector of smoothness parameters of length d 
            lengthscales (np.ndarray or torch.Tensor): vector of length scales of length d 
            scale (float): global scale 
            beta1 (np.ndarray or torch.Tensor): derivative orders with respect to the first argument 
            beta1 (np.ndarray or torch.Tensor): derivative orders with respect to the second argument
            torchify (bool): wheather or not to use PyTorch backend
        """
        super(KernelShiftInvar,self).__init__(dimension,alpha,lengthscales,scale,beta1,beta2,torchify)
        self.bpidxs = 2*self.alpha-self.beta1pbeta2
        if self.torchify:
            lgamma = self.npt.lgamma
        else:
            import scipy.special
            lgamma = scipy.special.loggamma
        self.const = (-1)**(self.alpha+self.beta2+1)*self.npt.exp(2*self.alpha*np.log(2*np.pi)-lgamma(self.bpidxs+1))
    def _low_call(self, x1, x2, k):
        delta = (x1[:,None,:]-x2[None,:,:])%1
        for j in range(self.d):
            k[:,:,j] = self.const[j]*bernoulli_poly(self.bpidxs[j].item(),delta[:,:,j])
    
class KernelDigShiftInvar(_ProdKernel):
    """
    Digitally Shift invariant kernel with product weights and derivatives 
    
    >>> d = 3
    >>> dnb2 = DigitalNetB2(d,randomize="LMS_DS",seed=7)
    >>> xb = dnb2.gen_samples(8,return_binary=True)
    >>> k = KernelDigShiftInvar(d,dnb2.t_lms)
    >>> with np.printoptions(precision=2):
    ...     k(xb,xb)
    array([[13.56,  0.17,  0.21,  0.85,  1.17,  0.28,  0.45,  0.43],
           [ 0.17, 13.56,  0.85,  0.21,  0.28,  1.17,  0.43,  0.45],
           [ 0.21,  0.85, 13.56,  0.17,  0.45,  0.43,  1.17,  0.28],
           [ 0.85,  0.21,  0.17, 13.56,  0.43,  0.45,  0.28,  1.17],
           [ 1.17,  0.28,  0.45,  0.43, 13.56,  0.17,  0.21,  0.85],
           [ 0.28,  1.17,  0.43,  0.45,  0.17, 13.56,  0.85,  0.21],
           [ 0.45,  0.43,  1.17,  0.28,  0.21,  0.85, 13.56,  0.17],
           [ 0.43,  0.45,  0.28,  1.17,  0.85,  0.21,  0.17, 13.56]])
    >>> import torch
    >>> kt = KernelDigShiftInvar(d,dnb2.t_lms,torchify=True)
    >>> kt(xb,xb[[0]])
    tensor([[13.5554],
            [ 0.1695],
            [ 0.2100],
            [ 0.8535],
            [ 1.1656],
            [ 0.2796],
            [ 0.4535],
            [ 0.4348]])
    """
    def __init__(self, dimension, t=None, alpha=3, lengthscales=1., scale=1., beta1=0, beta2=0, torchify=False):
        """
        Args:
            dimension (int): dimension of the input
            t (int): number of bits in each integer
            alpha (np.ndarray or torch.Tensor): vector of smoothness parameters of length d 
            lengthscales (np.ndarray or torch.Tensor): vector of length scales of length d 
            scale (float): global scale 
            beta1 (np.ndarray or torch.Tensor): derivative orders with respect to the first argument 
            beta1 (np.ndarray or torch.Tensor): derivative orders with respect to the second argument
            torchify (bool): wheather or not to use PyTorch backend
        """
        super(KernelDigShiftInvar,self).__init__(dimension,alpha,lengthscales,scale,beta1,beta2,torchify) 
        self.wfidxs = self.alpha-self.beta1pbeta2
        self.const = (-2)**(self.alpha-self.wfidxs)
        self.t = t
    def set_t(self, t):
        self.t = t
    def _low_call(self, x1, x2, k):
        delta = x1[:,None,:]^x2[None,:,:] # n1 x n2 x d
        for j in range(self.d):
            k[:,:,j] = self.const[j]*weighted_walsh_funcs(self.wfidxs[j].item(),delta[:,:,j],self.t)-self.ind[j]