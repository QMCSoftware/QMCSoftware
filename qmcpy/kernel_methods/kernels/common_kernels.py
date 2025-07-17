from .abstract_kernel import AbstractKernelScaleLengthscales
from .util import tf_exp_eps,tf_exp_eps_inv,tf_identity
from ...util import ParameterError
import numpy as np 

class KernelGaussian(AbstractKernelScaleLengthscales):
    """ 
    Gaussian / Squared Exponential kernel implemented using the product of exponentials. 
    """

    AUTOGRADKERNEL = True 

    def parsed___call__(self, x0, x1, beta0, beta1, batch_params):
        assert (beta0==0).all() and (beta1==0).all()
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        k = scale*self.npt.exp(-((x0-x1)/(np.sqrt(2)*lengthscales))**2).prod(-1)
        return k
    
class KernelSquaredExponential(AbstractKernelScaleLengthscales):
    """
    Gaussian / Squared Exponential kernel implemented using the pairwise distance function. 
    Please use KernelGaussian when using derivative information.
    """

    AUTOGRADKERNEL = True 

    def parsed___call__(self, x0, x1, beta0, beta1, batch_params):
        assert (beta0==0).all() and (beta1==0).all()
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        rdists = self.rel_pairwise_dist_func(x0,x1,lengthscales)
        k = scale*self.npt.exp(-rdists**2)
        return k

class KernelRationalQuadratic(AbstractKernelScaleLengthscales):

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
    
    def parsed___call__(self, x0, x1, beta0, beta1, batch_params):
        assert (beta0==0).all() and (beta1==0).all()
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        alpha = batch_params["alpha"][...,0]
        rdists = self.rel_pairwise_dist_func(x0,x1,lengthscales)
        k = scale*(1+rdists**2/alpha)**(-alpha)
        return k

class KernelMatern12(AbstractKernelScaleLengthscales):
    
    AUTOGRADKERNEL = True
    
    def parsed___call__(self, x0, x1, beta0, beta1, batch_params):
        assert (beta0==0).all() and (beta1==0).all()
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        rdists = self.rel_pairwise_dist_func(x0,x1,lengthscales)
        k = scale*self.npt.exp(-rdists)
        return k

class KernelMatern32(AbstractKernelScaleLengthscales):
    
    AUTOGRADKERNEL = True
    
    def parsed___call__(self, x0, x1, beta0, beta1, batch_params):
        assert (beta0==0).all() and (beta1==0).all()
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        rdists = self.rel_pairwise_dist_func(x0,x1,lengthscales)
        k = scale*(1+np.sqrt(3)*rdists)*self.npt.exp(-np.sqrt(3)*rdists)
        return k
    
class KernelMatern52(AbstractKernelScaleLengthscales):
    
    AUTOGRADKERNEL = True
    
    def parsed___call__(self, x0, x1, beta0, beta1, batch_params):
        assert (beta0==0).all() and (beta1==0).all()
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        rdists = self.rel_pairwise_dist_func(x0,x1,lengthscales)
        k = scale*((1+np.sqrt(5)*rdists+5*rdists**2/3)*self.npt.exp(-np.sqrt(5)*rdists))
        return k
        