import numpy as np

class _KernelProd(object):
    def __init__(self, dimension, lengthscales=1., scale=1., torchify=False, device="cpu", alpha=None, **kwargs):
        """
        Args:
            dimension (int): dimension of the input
            lengthscales (np.ndarray or torch.Tensor): vector of length scales of length d 
            scale (float): global scale 
            torchify (bool): wheather or not to use PyTorch backend
            alpha (np.ndarray or torch.Tensor): vector of smoothness parameters of length d 
        """
        self.d = dimension
        self.torchify = torchify 
        if self.torchify:
            import torch 
            self.npt = torch
            assert device!="mps", "mps does not (yet) support complex datatypes"
            self.ckwargs = {"device":device}
        else: 
            self.npt = np
            self.ckwargs = {}
        assert np.isscalar(lengthscales) or lengthscales.shape==(self.d,)
        assert np.isscalar(scale)
        if np.isscalar(lengthscales):
            lengthscales = lengthscales*self.npt.ones(self.d,dtype=float,**self.ckwargs)
        self.lengthscales = lengthscales
        self.scale = scale
        if alpha is None: alpha = self.DEFAULTALPHA
        assert alpha is np.nan or isinstance(alpha,int) or alpha.shape==(self.d,)
        if isinstance(alpha,int):
            alpha = alpha*self.npt.ones(self.d,dtype=int,**self.ckwargs)
        self.alpha = alpha
        self._my_init(**kwargs)
    def _my_init(self,**kwargs):
        pass
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
        if isinstance(beta1s,int): beta1s = beta1s*self.npt.ones(self.d,dtype=int,**self.ckwargs)
        if isinstance(beta2s,int): beta2s = beta2s*self.npt.ones(self.d,dtype=int,**self.ckwargs)
        beta1s = self.npt.atleast_2d(beta1s)
        beta2s = self.npt.atleast_2d(beta2s)
        m1 = len(beta1s) 
        m2 = len(beta2s)
        assert beta1s.shape==(m1,self.d) and beta2s.shape==(m2,self.d)
        if np.isscalar(c1s): c1s = c1s*self.npt.ones(m1,dtype=float,**self.ckwargs)
        if np.isscalar(c2s): c2s = c2s*self.npt.ones(m2,dtype=float,**self.ckwargs)
        assert c1s.shape==(m1,) and c2s.shape==(m2,)
        return self.parsed_call(x1,x2,n1,n2,beta1s,beta2s,m1,m2,c1s,c2s)
