from ...util import MethodImplementationError
import numpy as np 
from typing import Union


class AbstractKernel(object):

    def __init__(self, torchify):
        assert hasattr("self","d"), "Kernel object must have an attribute d, the dimension"
        assert hasattr("self","double_integral_01d"), "Kernel object must have an attribute double_integral_01d"
        if torchify: 
            import torch 
            self.npt = torch 
        else:
            self.npt = np

    def __call__(self, x0, x1, beta0=None, beta1=None):
        r"""
        Evaluate the kernel with (optional) partial derivatives 

        $$\partial_{\boldsymbol{x}_0}^{\boldsymbol{\beta}_0} \partial_{\boldsymbol{x}_1}^{\boldsymbol{\beta}_1} K^(\boldsymbol{x}_0,\boldsymbol{x}_1).$$
        
        Args:
            x0 (Union[np.ndarray,torch.Tensor]): Shape `x0.shape=(...,d)` first input to kernel with 
            x1 (Union[np.ndarray,torch.Tensor]): Shape `x1.shape=(...,d)` second input to kernel with 
            beta0 (Union[np.ndarray,torch.Tensor]): `beta0.shape==(d,)` derivative orders with respect to first inputs, $\boldsymbol{\beta}_0$.
            beta1 (Union[np.ndarray,torch.Tensor]): `beta1.shape==(d,)` derivative orders with respect to first inputs, $\boldsymbol{\beta}_1$.
        
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
        assert beta0.shape==(self,self.d), "expected beta0.shape=(%d,) but got beta0.shape=%s"%(self.d,str(tuple(beta0.shape)))
        assert beta1.shape==(self,self.d), "expected beta1.shape=(%d,) but got beta1.shape=%s"%(self.d,str(tuple(beta1.shape)))
        assert (beta0%1==0).all() and (beta0>=0).all(), "require int beta0 >= 0"
        assert (beta1%1==0).all() and (beta1>=0).all(), "require int beta1 >= 0"
        return self.parsed___call__(x0,x1,beta0,beta1)
    
    def parsed___call__(self, x0, x1, beta0, beta1):
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
