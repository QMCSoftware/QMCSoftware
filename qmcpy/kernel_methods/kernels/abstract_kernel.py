from ...util import MethodImplementationError
import numpy as np 
from typing import Union
from .util import tf_exp_eps,tf_exp_eps_inv,parse_assign_param,tf_identity


class AbstractKernel(object):

    def __new__(cls, *args, **kwargs):
        if "torchify" in kwargs and kwargs["torchify"]:
            import torch 
            x = type(cls.__name__,(cls,torch.nn.Module),{})
            instance = super().__new__(x)
        else:
            instance = super().__new__(cls)
        return instance

    def __init__(self, d, shape_batch, torchify, device):
        super().__init__()
        # dimension 
        assert d%1==0 and d>0, "dimension d must be a positive int"
        self.d = d
        # shape_batch 
        if isinstance(shape_batch,int): shape_batch = [shape_batch]
        assert isinstance(shape_batch,(list,tuple))
        self.shape_batch = list(shape_batch)
        self.ndim_batch = len(self.shape_batch)
        # torchify
        self.torchify = torchify 
        if self.torchify: 
            import torch 
            self.npt = torch 
            self.device = torch.device(device) 
            self.nptkwargs = {"device":device}
        else:
            self.npt = np
            self.nptkwargs = {}
        self.batch_params = {}
    
    def __call__(self, x0, x1, beta0=None, beta1=None):
        r"""
        Evaluate the kernel with (optional) partial derivatives 

        $$\partial_{\boldsymbol{x}_0}^{\boldsymbol{\beta}_0} \partial_{\boldsymbol{x}_1}^{\boldsymbol{\beta}_1} K^(\boldsymbol{x}_0,\boldsymbol{x}_1).$$
        
        Args:
            x0 (Union[np.ndarray,torch.Tensor]): Shape `x0.shape=(...,d)` first input to kernel with 
            x1 (Union[np.ndarray,torch.Tensor]): Shape `x1.shape=(...,d)` second input to kernel with 
            beta0 (Union[np.ndarray,torch.Tensor]): Shape `beta0.shape==(d,)` derivative orders with respect to first inputs, $\boldsymbol{\beta}_0$.
            beta1 (Union[np.ndarray,torch.Tensor]): Shape `beta1.shape==(d,)` derivative orders with respect to first inputs, $\boldsymbol{\beta}_1$.
        
        Returns:
            k (Union[np.ndarray,torch.Tensor]): Shape `y.shape=(x0+x1).shape[:-1]` kernel evaluations. 
        """
        if self.npt==np:
            assert isinstance(x0,np.ndarray) 
            assert isinstance(x0,np.ndarray) 
        else: # self.npt==torch
            assert isinstance(x0,self.npt.Tensor)
            assert isinstance(x1,self.npt.Tensor)
        assert x0.shape[-1]==self.d, "the size of the last dimension of x0 must equal d=%d, got x0.shape=%s"%(self.d,str(tuple(x0.shape)))
        assert x1.shape[-1]==self.d, "the size of the last dimension of x1 must equal d=%d, got x1.shape=%s"%(self.d,str(tuple(x1.shape)))
        if beta0 is None:
            beta0 = self.npt.zeros(self.d,dtype=int)
        if beta1 is None:
            beta1 = self.npt.zeros(self.d,dtype=int)
        beta0 = self.npt.atleast_1d(beta0)
        beta1 = self.npt.atleast_1d(beta1)
        assert beta0.shape==(self.d,), "expected beta0.shape=(%d,) but got beta0.shape=%s"%(self.d,str(tuple(beta0.shape)))
        assert beta1.shape==(self.d,), "expected beta1.shape=(%d,) but got beta1.shape=%s"%(self.d,str(tuple(beta1.shape)))
        assert (beta0%1==0).all() and (beta0>=0).all(), "require int beta0 >= 0"
        assert (beta1%1==0).all() and (beta1>=0).all(), "require int beta1 >= 0"
        kndimones = [1]*max(len(x0.shape)-1,len(x1.shape)-1)
        batch_params = {pname: batch_param.reshape(list(batch_param.shape[:-1])+kndimones+[int(batch_param.shape[-1])]) for pname,batch_param in self.batch_params.items()}
        return self.parsed___call__(x0,x1,beta0,beta1,batch_params)
    
    def parsed___call__(self, x0, x1, beta0, beta1, batch_params):
        raise MethodImplementationError(self, 'parsed___call__')

    def single_integral_01d(self, x):
        r"""
        Evaluate the integral of the kernel over the unit cube

        $$\tilde{K}(\boldsymbol{x}) = \int_{[0,1]^d} K(\boldsymbol{x},\boldsymbol{z}) \mathrm{d} \boldsymbol{z}.$$
        
        Args:
            x (Union[np.ndarray,torch.Tensor]): Shape `x0.shape=(...,d)` first input to kernel with 
        
        Returns:
            tildek (Union[np.ndarray,torch.Tensor]): Shape `y.shape=x.shape[:-1]` integral kernel evaluations. 
        """
        if self.npt==np:
            assert isinstance(x,np.ndarray) 
        else: # self.npt==torch
            assert isinstance(x,self.npt.Tensor)
        assert x.shape[-1]==self.d, "the size of the last dimension of x must equal d=%d, got x.shape=%s"%(self.d,str(tuple(x.shape)))
        return self.parsed_single_integral_01d(x)
    
    def parsed_single_integral_01d(self, x):
        raise MethodImplementationError(self, 'parsed_single_integral_01d')
    
    @property
    def double_integral_01d(self):
        raise MethodImplementationError(self, 'double_integral_01d')

    def rel_pairwise_dist_func(self, x0, x1, lengthscales):
        return self.npt.linalg.norm((x0-x1)/(np.sqrt(2)*lengthscales),ord=2,dim=-1)
    
    def parse_assign_param(self,
            pname, 
            param, 
            shape_param, 
            requires_grad_param,
            tfs_param,
            endsize_ops,
            constraints):
        return parse_assign_param(
            pname = pname,
            param = param, 
            shape_param = shape_param,
            requires_grad_param = requires_grad_param,
            tfs_param = tfs_param,
            endsize_ops = endsize_ops,
            constraints = constraints,
            shape_batch = self.shape_batch,
            torchify = self.torchify,
            npt = self.npt, 
            nptkwargs = self.nptkwargs)
    
class AbstractKernelScaleLengthscales(AbstractKernel):

    def __init__(self,
            d, 
            scale = 1., 
            lengthscales = 1.,
            shape_batch = [],
            shape_scale = [1],
            shape_lengthscales = None, 
            tfs_scale = (tf_exp_eps_inv,tf_exp_eps),
            tfs_lengthscales = (tf_exp_eps_inv,tf_exp_eps),
            torchify = False, 
            requires_grad_scale = True, 
            requires_grad_lengthscales = True, 
            device = "cpu",
            ):
        super().__init__(d=d,shape_batch=shape_batch,torchify=torchify,device=device)
        self.raw_scale,self.tf_scale = self.parse_assign_param(
            pname = "scale",
            param = scale, 
            shape_param = shape_scale,
            requires_grad_param = requires_grad_scale,
            tfs_param = tfs_scale,
            endsize_ops = [1],
            constraints = ["POSITIVE"])
        self.batch_params["scale"] = self.scale
        self.raw_lengthscales,self.tf_lengthscales = self.parse_assign_param(
            pname = "lengthscales",
            param = lengthscales, 
            shape_param = [self.d] if shape_lengthscales is None else [],
            requires_grad_param = requires_grad_lengthscales,
            tfs_param = tfs_lengthscales,
            endsize_ops = [1,self.d],
            constraints = ["POSITIVE"])
        self.batch_params["lengthscales"] = self.lengthscales

    @property
    def scale(self):
        return self.tf_scale(self.raw_scale)
    
    @property
    def lengthscales(self):
        return self.tf_lengthscales(self.raw_lengthscales)
        