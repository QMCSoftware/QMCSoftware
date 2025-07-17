from .abstract_kernel import AbstractKernelScaleLengthscales
from .util import tf_exp_eps,tf_exp_eps_inv,tf_identity
from ..util import ParameterError
import numpy as np 

class KernelGaussian(AbstractKernelScaleLengthscales):
    
    r""" 
    Gaussian / Squared Exponential kernel implemented using the product of exponentials. 

    Examples:
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelGaussian(d=2)
        >>> x = rng.uniform(low=0,high=1,size=(4,2))
        >>> kernel(x[0],x[0])
        1.0
        >>> kernel(x,x)
        array([1., 1., 1., 1.])
        >>> kernel(x[:,None,:],x[None,:,:])
        array([[1.        , 0.78888466, 0.9483142 , 0.8228498 ],
               [0.78888466, 1.        , 0.72380317, 0.62226176],
               [0.9483142 , 0.72380317, 1.        , 0.95613874],
               [0.8228498 , 0.62226176, 0.95613874, 1.        ]])
        
        Multiple randomizations 
        
        >>> x = rng.uniform(low=0,high=1,size=(6,5,2))
        >>> kernel(x,x).shape 
        (6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (6, 5, 5)
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        (6, 6, 5, 5)

        Batch hyperparameters 

        >>> kernel = KernelGaussian(
        ...     d = 2, 
        ...     shape_batch = [5,4,3],
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,1])
        >>> kernel(x,x).shape 
        (4, 3, 6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (4, 3, 6, 5, 5)
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        (4, 3, 6, 6, 5, 5)

        PyTorch

        >>> import torch
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelGaussian(d=2,torchify=True)
        >>> x = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,2)))
        >>> kernel(x[0],x[0])
        tensor(1., dtype=torch.float64, grad_fn=<MulBackward0>)
        >>> kernel(x,x)
        tensor([1., 1., 1., 1.], dtype=torch.float64, grad_fn=<MulBackward0>)
        >>> kernel(x[:,None,:],x[None,:,:])
        tensor([[1.0000, 0.7889, 0.9483, 0.8228],
                [0.7889, 1.0000, 0.7238, 0.6223],
                [0.9483, 0.7238, 1.0000, 0.9561],
                [0.8228, 0.6223, 0.9561, 1.0000]], dtype=torch.float64,
               grad_fn=<MulBackward0>)
        
        Multiple randomizations 
        
        >>> x = torch.from_numpy(rng.uniform(low=0,high=1,size=(6,5,2)))
        >>> kernel(x,x).shape 
        torch.Size([6, 5])
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        torch.Size([6, 5, 5])
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        torch.Size([6, 6, 5, 5])

        Batch hyperparameters 

        >>> kernel = KernelGaussian(
        ...     d = 2, 
        ...     shape_batch = [5,4,3],
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,1],
        ...     torchify = True)
        >>> kernel(x,x).shape 
        torch.Size([4, 3, 6, 5])
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        torch.Size([4, 3, 6, 5, 5])
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        torch.Size([4, 3, 6, 6, 5, 5])
    """

    AUTOGRADKERNEL = True 

    def parsed___call__(self, x0, x1, batch_params):
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        k = scale*self.npt.exp(-((x0-x1)/(np.sqrt(2)*lengthscales))**2).prod(-1)
        return k
    
class KernelSquaredExponential(AbstractKernelScaleLengthscales):
    
    r"""
    Gaussian / Squared Exponential kernel implemented using the pairwise distance function. 
    Please use `KernelGaussian` when using derivative information.
    
    Examples:
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelSquaredExponential(d=2)
        >>> x = rng.uniform(low=0,high=1,size=(4,2))
        >>> kernel(x[0],x[0])
        1.0
        >>> kernel(x,x)
        array([1., 1., 1., 1.])
        >>> kernel(x[:,None,:],x[None,:,:])
        array([[1.        , 0.78888466, 0.9483142 , 0.8228498 ],
               [0.78888466, 1.        , 0.72380317, 0.62226176],
               [0.9483142 , 0.72380317, 1.        , 0.95613874],
               [0.8228498 , 0.62226176, 0.95613874, 1.        ]])
        
        Multiple randomizations 
        
        >>> x = rng.uniform(low=0,high=1,size=(6,5,2))
        >>> kernel(x,x).shape 
        (6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (6, 5, 5)
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        (6, 6, 5, 5)

        Batch hyperparameters 

        >>> kernel = KernelSquaredExponential(
        ...     d = 2, 
        ...     shape_batch = [5,4,3],
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,1])
        >>> kernel(x,x).shape 
        (4, 3, 6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (4, 3, 6, 5, 5)
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        (4, 3, 6, 6, 5, 5)

        PyTorch

        >>> import torch
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelSquaredExponential(d=2,torchify=True)
        >>> x = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,2)))
        >>> kernel(x[0],x[0])
        tensor(1., dtype=torch.float64, grad_fn=<MulBackward0>)
        >>> kernel(x,x)
        tensor([1., 1., 1., 1.], dtype=torch.float64, grad_fn=<MulBackward0>)
        >>> kernel(x[:,None,:],x[None,:,:])
        tensor([[1.0000, 0.7889, 0.9483, 0.8228],
                [0.7889, 1.0000, 0.7238, 0.6223],
                [0.9483, 0.7238, 1.0000, 0.9561],
                [0.8228, 0.6223, 0.9561, 1.0000]], dtype=torch.float64,
               grad_fn=<MulBackward0>)
        
        Multiple randomizations 
        
        >>> x = torch.from_numpy(rng.uniform(low=0,high=1,size=(6,5,2)))
        >>> kernel(x,x).shape 
        torch.Size([6, 5])
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        torch.Size([6, 5, 5])
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        torch.Size([6, 6, 5, 5])

        Batch hyperparameters 

        >>> kernel = KernelSquaredExponential(
        ...     d = 2, 
        ...     shape_batch = [5,4,3],
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,1],
        ...     torchify = True)
        >>> kernel(x,x).shape 
        torch.Size([4, 3, 6, 5])
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        torch.Size([4, 3, 6, 5, 5])
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        torch.Size([4, 3, 6, 6, 5, 5])
    """

    AUTOGRADKERNEL = True 

    def parsed___call__(self, x0, x1, batch_params):
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        rdists = self.rel_pairwise_dist_func(x0,x1,lengthscales)
        k = scale*self.npt.exp(-rdists**2)
        return k

class KernelRationalQuadratic(AbstractKernelScaleLengthscales):

    r"""
    Rational Quadratic kernel 

    Examples:
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelRationalQuadratic(d=2)
        >>> x = rng.uniform(low=0,high=1,size=(4,2))
        >>> kernel(x[0],x[0])
        1.0
        >>> kernel(x,x)
        array([1., 1., 1., 1.])
        >>> kernel(x[:,None,:],x[None,:,:])
        array([[1.        , 0.80831912, 0.94960504, 0.83683297],
               [0.80831912, 1.        , 0.75572321, 0.67824456],
               [0.94960504, 0.75572321, 1.        , 0.95707312],
               [0.83683297, 0.67824456, 0.95707312, 1.        ]])
        
        Multiple randomizations 
        
        >>> x = rng.uniform(low=0,high=1,size=(6,5,2))
        >>> kernel(x,x).shape 
        (6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (6, 5, 5)
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        (6, 6, 5, 5)

        Batch hyperparameters 

        >>> kernel = KernelRationalQuadratic(
        ...     d = 2, 
        ...     shape_batch = [5,4,3],
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,1])
        >>> kernel(x,x).shape 
        (4, 3, 6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (4, 3, 6, 5, 5)
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        (4, 3, 6, 6, 5, 5)

        PyTorch

        >>> import torch
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelRationalQuadratic(d=2,torchify=True)
        >>> x = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,2)))
        >>> kernel(x[0],x[0])
        tensor(1., dtype=torch.float64, grad_fn=<MulBackward0>)
        >>> kernel(x,x)
        tensor([1., 1., 1., 1.], dtype=torch.float64, grad_fn=<MulBackward0>)
        >>> kernel(x[:,None,:],x[None,:,:])
        tensor([[1.0000, 0.8083, 0.9496, 0.8368],
                [0.8083, 1.0000, 0.7557, 0.6782],
                [0.9496, 0.7557, 1.0000, 0.9571],
                [0.8368, 0.6782, 0.9571, 1.0000]], dtype=torch.float64,
               grad_fn=<MulBackward0>)
        
        Multiple randomizations 
        
        >>> x = torch.from_numpy(rng.uniform(low=0,high=1,size=(6,5,2)))
        >>> kernel(x,x).shape 
        torch.Size([6, 5])
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        torch.Size([6, 5, 5])
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        torch.Size([6, 6, 5, 5])

        Batch hyperparameters 

        >>> kernel = KernelRationalQuadratic(
        ...     d = 2, 
        ...     shape_batch = [5,4,3],
        ...     shape_alpha = [4,3,1],
        ...     shape_lengthscales = [3,1],
        ...     torchify = True)
        >>> kernel(x,x).shape 
        torch.Size([4, 3, 6, 5])
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        torch.Size([4, 3, 6, 5, 5])
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        torch.Size([4, 3, 6, 6, 5, 5])
    """
    
    AUTOGRADKERNEL = True

    def __init__(self,
            d, 
            scale = 1., 
            lengthscales = 1.,
            alpha = 1.,
            shape_batch = [],
            shape_scale = [1],
            shape_lengthscales = None, 
            shape_alpha = [1],
            tfs_scale = (tf_exp_eps_inv,tf_exp_eps),
            tfs_lengthscales = (tf_exp_eps_inv,tf_exp_eps),
            tfs_alphas = (tf_exp_eps_inv,tf_exp_eps),
            torchify = False, 
            requires_grad_scale = True, 
            requires_grad_lengthscales = True, 
            requires_grad_alpha = True, 
            device = "cpu",
            ):
        super().__init__(
            d = d, 
            scale = scale, 
            lengthscales = lengthscales,
            shape_batch = shape_batch,
            shape_scale = shape_scale,
            shape_lengthscales = shape_lengthscales, 
            tfs_scale = tfs_scale,
            tfs_lengthscales = tfs_lengthscales,
            torchify = torchify, 
            requires_grad_scale = requires_grad_scale, 
            requires_grad_lengthscales = requires_grad_lengthscales, 
            device = device,
        )
        self.raw_alpha,self.tf_alpha = self.parse_assign_param(
            pname = "alpha",
            param = alpha, 
            shape_param = shape_alpha,
            requires_grad_param = requires_grad_alpha,
            tfs_param = tfs_alphas,
            endsize_ops = [1],
            constraints = ["POSITIVE"])
        self.batch_params["alpha"] = self.alpha
    
    @property
    def alpha(self):
        return self.tf_alpha(self.raw_alpha)
    
    def parsed___call__(self, x0, x1, batch_params):
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        alpha = batch_params["alpha"][...,0]
        rdists = self.rel_pairwise_dist_func(x0,x1,lengthscales)
        k = scale*(1+rdists**2/alpha)**(-alpha)
        return k

class KernelMatern12(AbstractKernelScaleLengthscales):
    
    r""" 
    Matern kernel with $\alpha=1/2$. 

    Examples:
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelMatern12(d=2)
        >>> x = rng.uniform(low=0,high=1,size=(4,2))
        >>> kernel(x[0],x[0])
        1.0
        >>> kernel(x,x)
        array([1., 1., 1., 1.])
        >>> kernel(x[:,None,:],x[None,:,:])
        array([[1.        , 0.61448839, 0.79424131, 0.64302787],
               [0.61448839, 1.        , 0.56635268, 0.50219691],
               [0.79424131, 0.56635268, 1.        , 0.80913986],
               [0.64302787, 0.50219691, 0.80913986, 1.        ]])
        
        Multiple randomizations 
        
        >>> x = rng.uniform(low=0,high=1,size=(6,5,2))
        >>> kernel(x,x).shape 
        (6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (6, 5, 5)
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        (6, 6, 5, 5)

        Batch hyperparameters 

        >>> kernel = KernelMatern12(
        ...     d = 2, 
        ...     shape_batch = [5,4,3],
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,1])
        >>> kernel(x,x).shape 
        (4, 3, 6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (4, 3, 6, 5, 5)
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        (4, 3, 6, 6, 5, 5)

        PyTorch

        >>> import torch
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelMatern12(d=2,torchify=True)
        >>> x = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,2)))
        >>> kernel(x[0],x[0])
        tensor(1., dtype=torch.float64, grad_fn=<MulBackward0>)
        >>> kernel(x,x)
        tensor([1., 1., 1., 1.], dtype=torch.float64, grad_fn=<MulBackward0>)
        >>> kernel(x[:,None,:],x[None,:,:])
        tensor([[1.0000, 0.6145, 0.7942, 0.6430],
                [0.6145, 1.0000, 0.5664, 0.5022],
                [0.7942, 0.5664, 1.0000, 0.8091],
                [0.6430, 0.5022, 0.8091, 1.0000]], dtype=torch.float64,
               grad_fn=<MulBackward0>)
        
        Multiple randomizations 
        
        >>> x = torch.from_numpy(rng.uniform(low=0,high=1,size=(6,5,2)))
        >>> kernel(x,x).shape 
        torch.Size([6, 5])
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        torch.Size([6, 5, 5])
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        torch.Size([6, 6, 5, 5])

        Batch hyperparameters 

        >>> kernel = KernelMatern12(
        ...     d = 2, 
        ...     shape_batch = [5,4,3],
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,1],
        ...     torchify = True)
        >>> kernel(x,x).shape 
        torch.Size([4, 3, 6, 5])
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        torch.Size([4, 3, 6, 5, 5])
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        torch.Size([4, 3, 6, 6, 5, 5])
    """
    
    AUTOGRADKERNEL = True
    
    def parsed___call__(self, x0, x1, batch_params):
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        rdists = self.rel_pairwise_dist_func(x0,x1,lengthscales)
        k = scale*self.npt.exp(-rdists)
        return k

class KernelMatern32(AbstractKernelScaleLengthscales):
    
    r""" 
    Matern kernel with $\alpha=3/2$. 

    Examples:
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelMatern32(d=2)
        >>> x = rng.uniform(low=0,high=1,size=(4,2))
        >>> kernel(x[0],x[0])
        1.0
        >>> kernel(x,x)
        array([1., 1., 1., 1.])
        >>> kernel(x[:,None,:],x[None,:,:])
        array([[1.        , 0.79309639, 0.93871358, 0.82137958],
               [0.79309639, 1.        , 0.74137353, 0.66516872],
               [0.93871358, 0.74137353, 1.        , 0.9471166 ],
               [0.82137958, 0.66516872, 0.9471166 , 1.        ]])
        
        Multiple randomizations 
        
        >>> x = rng.uniform(low=0,high=1,size=(6,5,2))
        >>> kernel(x,x).shape 
        (6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (6, 5, 5)
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        (6, 6, 5, 5)

        Batch hyperparameters 

        >>> kernel = KernelMatern32(
        ...     d = 2, 
        ...     shape_batch = [5,4,3],
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,1])
        >>> kernel(x,x).shape 
        (4, 3, 6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (4, 3, 6, 5, 5)
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        (4, 3, 6, 6, 5, 5)

        PyTorch

        >>> import torch
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelMatern32(d=2,torchify=True)
        >>> x = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,2)))
        >>> kernel(x[0],x[0])
        tensor(1., dtype=torch.float64, grad_fn=<MulBackward0>)
        >>> kernel(x,x)
        tensor([1., 1., 1., 1.], dtype=torch.float64, grad_fn=<MulBackward0>)
        >>> kernel(x[:,None,:],x[None,:,:])
        tensor([[1.0000, 0.7931, 0.9387, 0.8214],
                [0.7931, 1.0000, 0.7414, 0.6652],
                [0.9387, 0.7414, 1.0000, 0.9471],
                [0.8214, 0.6652, 0.9471, 1.0000]], dtype=torch.float64,
               grad_fn=<MulBackward0>)
        
        Multiple randomizations 
        
        >>> x = torch.from_numpy(rng.uniform(low=0,high=1,size=(6,5,2)))
        >>> kernel(x,x).shape 
        torch.Size([6, 5])
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        torch.Size([6, 5, 5])
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        torch.Size([6, 6, 5, 5])

        Batch hyperparameters 

        >>> kernel = KernelMatern32(
        ...     d = 2, 
        ...     shape_batch = [5,4,3],
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,1],
        ...     torchify = True)
        >>> kernel(x,x).shape 
        torch.Size([4, 3, 6, 5])
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        torch.Size([4, 3, 6, 5, 5])
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        torch.Size([4, 3, 6, 6, 5, 5])
    """
    
    AUTOGRADKERNEL = True
    
    def parsed___call__(self, x0, x1, batch_params):
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        rdists = self.rel_pairwise_dist_func(x0,x1,lengthscales)
        k = scale*(1+np.sqrt(3)*rdists)*self.npt.exp(-np.sqrt(3)*rdists)
        return k
    
class KernelMatern52(AbstractKernelScaleLengthscales):
    
    r""" 
    Matern kernel with $\alpha=5/2$. 

    Examples:
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelMatern52(d=2)
        >>> x = rng.uniform(low=0,high=1,size=(4,2))
        >>> kernel(x[0],x[0])
        1.0
        >>> kernel(x,x)
        array([1., 1., 1., 1.])
        >>> kernel(x[:,None,:],x[None,:,:])
        array([[1.        , 0.83612941, 0.95801903, 0.861472  ],
               [0.83612941, 1.        , 0.78812397, 0.71396963],
               [0.95801903, 0.78812397, 1.        , 0.96425994],
               [0.861472  , 0.71396963, 0.96425994, 1.        ]])
        
        Multiple randomizations 
        
        >>> x = rng.uniform(low=0,high=1,size=(6,5,2))
        >>> kernel(x,x).shape 
        (6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (6, 5, 5)
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        (6, 6, 5, 5)

        Batch hyperparameters 

        >>> kernel = KernelMatern52(
        ...     d = 2, 
        ...     shape_batch = [5,4,3],
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,1])
        >>> kernel(x,x).shape 
        (4, 3, 6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (4, 3, 6, 5, 5)
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        (4, 3, 6, 6, 5, 5)

        PyTorch

        >>> import torch
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelMatern52(d=2,torchify=True)
        >>> x = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,2)))
        >>> kernel(x[0],x[0])
        tensor(1., dtype=torch.float64, grad_fn=<MulBackward0>)
        >>> kernel(x,x)
        tensor([1., 1., 1., 1.], dtype=torch.float64, grad_fn=<MulBackward0>)
        >>> kernel(x[:,None,:],x[None,:,:])
        tensor([[1.0000, 0.8361, 0.9580, 0.8615],
                [0.8361, 1.0000, 0.7881, 0.7140],
                [0.9580, 0.7881, 1.0000, 0.9643],
                [0.8615, 0.7140, 0.9643, 1.0000]], dtype=torch.float64,
               grad_fn=<MulBackward0>)
        
        Multiple randomizations 
        
        >>> x = torch.from_numpy(rng.uniform(low=0,high=1,size=(6,5,2)))
        >>> kernel(x,x).shape 
        torch.Size([6, 5])
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        torch.Size([6, 5, 5])
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        torch.Size([6, 6, 5, 5])

        Batch hyperparameters 

        >>> kernel = KernelMatern52(
        ...     d = 2, 
        ...     shape_batch = [5,4,3],
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,1],
        ...     torchify = True)
        >>> kernel(x,x).shape 
        torch.Size([4, 3, 6, 5])
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        torch.Size([4, 3, 6, 5, 5])
        >>> kernel(x[:,None,:,None,:],x[None,:,None,:,:]).shape
        torch.Size([4, 3, 6, 6, 5, 5])
    """
    
    AUTOGRADKERNEL = True
    
    def parsed___call__(self, x0, x1, batch_params):
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        rdists = self.rel_pairwise_dist_func(x0,x1,lengthscales)
        k = scale*((1+np.sqrt(5)*rdists+5*rdists**2/3)*self.npt.exp(-np.sqrt(5)*rdists))
        return k
        
