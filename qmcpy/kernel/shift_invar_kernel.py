from .abstract_kernel import AbstractKernelScaleLengthscales
from ..util.transforms import tf_exp_eps,tf_exp_eps_inv,tf_identity
from ..util.shift_invar_ops import BERNOULLIPOLYSDICT,bernoulli_poly
from ..util import ParameterError
import numpy as np 
from typing import Union,Tuple
import scipy.special


class KernelShiftInvar(AbstractKernelScaleLengthscales):
    r""" 
    Shift invariant kernel with 
    smoothness $\boldsymbol{\alpha}$, product weights (lengthscales) $\boldsymbol{\gamma}$, and scale $S$:
    
    $$\begin{aligned}
        K(\boldsymbol{x},\boldsymbol{z}) &= S \prod_{j=1}^d \left(1+ \gamma_j \tilde{K}_{\alpha_j}((x_j - z_j) \mod 1))\right), \\ 
        \tilde{K}_\alpha(x) &= (-1)^{\alpha+1}\frac{(2 \pi)^{2 \alpha}}{(2\alpha)!} B_{2\alpha}(x)
    \end{aligned}$$

    where $B_n$ is the $n^\text{th}$ Bernoulli polynomial.
    
    Examples:
        >>> from qmcpy import Lattice, fftbr, ifftbr
        >>> n = 8
        >>> d = 4
        >>> lat = Lattice(d,seed=11)
        >>> x = lat(n)
        >>> x.shape
        (8, 4)
        >>> x.dtype
        dtype('float64')
        >>> kernel = KernelShiftInvar(
        ...     d = d, 
        ...     alpha = list(range(1,d+1)),
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)])
        >>> k00 = kernel(x[0],x[0])
        >>> k00.item()
        91.23444453396341
        >>> k0 = kernel(x,x[0])
        >>> with np.printoptions(precision=2):
        ...     print(k0)
        [91.23 -2.32  5.69  5.69 12.7  -4.78 -4.78 12.7 ]
        >>> assert k0[0]==k00
        >>> kmat = kernel(x[:,None,:],x[None,:,:])
        >>> with np.printoptions(precision=2):
        ...     print(kmat)
        [[91.23 -2.32  5.69  5.69 12.7  -4.78 -4.78 12.7 ]
         [-2.32 91.23  5.69  5.69 -4.78 12.7  12.7  -4.78]
         [ 5.69  5.69 91.23 -2.32 12.7  -4.78 12.7  -4.78]
         [ 5.69  5.69 -2.32 91.23 -4.78 12.7  -4.78 12.7 ]
         [12.7  -4.78 12.7  -4.78 91.23 -2.32  5.69  5.69]
         [-4.78 12.7  -4.78 12.7  -2.32 91.23  5.69  5.69]
         [-4.78 12.7  12.7  -4.78  5.69  5.69 91.23 -2.32]
         [12.7  -4.78 -4.78 12.7   5.69  5.69 -2.32 91.23]]
        >>> assert (kmat[:,0]==k0).all()
        >>> lam = np.sqrt(n)*fftbr(k0)
        >>> y = np.random.Generator(np.random.PCG64(7)).uniform(low=0,high=1,size=(n))
        >>> np.allclose(ifftbr(fftbr(y)*lam),kmat@y)
        True
        >>> np.allclose(ifftbr(fftbr(y)/lam),np.linalg.solve(kmat,y))
        True
        >>> import torch 
        >>> xtorch = torch.from_numpy(x)
        >>> kernel_torch = KernelShiftInvar(
        ...     d = d, 
        ...     alpha = list(range(1,d+1)),
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)],
        ...     torchify = True)
        >>> kmat_torch = kernel_torch(xtorch[:,None,:],xtorch[None,:,:])
        >>> np.allclose(kmat_torch.detach().numpy(),kmat)
        True
        >>> kernel.single_integral_01d(x)
        array([10.])
        >>> kernel_torch.single_integral_01d(xtorch)
        tensor([10.], grad_fn=<SelectBackward0>)

        Derivatives 

        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> scale = rng.uniform(low=0,high=1,size=(1,))
        >>> lengthscales = rng.uniform(low=0,high=1,size=(3,))
        >>> kernel = KernelShiftInvar(
        ...     d = 3,
        ...     alpha = 3,
        ...     torchify = True,
        ...     scale = torch.from_numpy(scale),
        ...     lengthscales = torch.from_numpy(lengthscales))
        >>> x0 = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,))).requires_grad_(True)
        >>> x1 = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,))).requires_grad_(True)
        >>> x2 = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,))).requires_grad_(True)
        >>> x = torch.stack([x0,x1,x2],axis=-1)
        >>> z0 = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,))).requires_grad_(True)
        >>> z1 = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,))).requires_grad_(True)
        >>> z2 = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,))).requires_grad_(True)
        >>> z = torch.stack([z0,z1,z2],axis=-1)
        >>> c = torch.from_numpy(rng.uniform(low=0,high=1,size=(2,)))
        >>> beta0 = torch.tensor([
        ...     [1,0,0],
        ...     [0,2,0]])
        >>> beta1 = torch.tensor([
        ...     [0,0,2],
        ...     [2,1,0]])
        >>> with torch.no_grad():
        ...     y = kernel(x,z,beta0,beta1,c)
        >>> y
        tensor([1455.1395, 9475.5695, 7807.0759, 2785.4733], dtype=torch.float64)
        >>> y_no_deriv = kernel(x,z)
        >>> y_first = y_no_deriv.clone()
        >>> y_first = torch.autograd.grad(y_first,x0,grad_outputs=torch.ones_like(y_first,requires_grad=True),create_graph=True)[0]
        >>> y_first = torch.autograd.grad(y_first,z2,grad_outputs=torch.ones_like(y_first,requires_grad=True),create_graph=True)[0]
        >>> y_first = torch.autograd.grad(y_first,z2,grad_outputs=torch.ones_like(y_first,requires_grad=True),create_graph=True)[0]
        >>> y_second = y_no_deriv.clone()
        >>> y_second = torch.autograd.grad(y_second,x1,grad_outputs=torch.ones_like(y_second,requires_grad=True),create_graph=True)[0]
        >>> y_second = torch.autograd.grad(y_second,x1,grad_outputs=torch.ones_like(y_second,requires_grad=True),create_graph=True)[0]
        >>> y_second = torch.autograd.grad(y_second,z0,grad_outputs=torch.ones_like(y_second,requires_grad=True),create_graph=True)[0]
        >>> y_second = torch.autograd.grad(y_second,z0,grad_outputs=torch.ones_like(y_second,requires_grad=True),create_graph=True)[0]
        >>> y_second = torch.autograd.grad(y_second,z1,grad_outputs=torch.ones_like(y_second,requires_grad=True),create_graph=True)[0]
        >>> yhat = (y_first*c[0]+y_second*c[1]).detach()
        >>> yhat
        tensor([1455.1396, 9475.5700, 7807.0762, 2785.4735], dtype=torch.float64)
        >>> torch.allclose(y,yhat)
        True
        >>> kernel = KernelShiftInvar(
        ...     d = 3,
        ...     alpha = 3,
        ...     scale = scale,
        ...     lengthscales = lengthscales)
        >>> ynp = kernel(x.detach().numpy(),z.detach().numpy(),beta0.numpy(),beta1.numpy(),c.numpy())
        >>> ynp
        array([1455.14170759, 9475.58534891, 7807.088978  , 2785.47661877])
        >>> np.allclose(ynp,y.numpy())
        True
        
    **References:** 
    
    1.  Kaarnioja, Vesa, Frances Y. Kuo, and Ian H. Sloan.  
        "Lattice-based kernel approximation and serendipitous weights for parametric PDEs in very high dimensions."  
        International Conference on Monte Carlo and Quasi-Monte Carlo Methods in Scientific Computing. Cham: Springer International Publishing, 2022.
    """
    
    AUTOGRADKERNEL = False
    
    def __init__(self,
            d, 
            scale = 1., 
            lengthscales = 1.,
            alpha = 2,
            shape_scale = [1],
            shape_lengthscales = None, 
            tfs_scale = (tf_exp_eps_inv,tf_exp_eps),
            tfs_lengthscales = (tf_exp_eps_inv,tf_exp_eps),
            torchify = False, 
            requires_grad_scale = True, 
            requires_grad_lengthscales = True, 
            device = "cpu",
            compile_call = False,
            comiple_call_kwargs = {},
            ):
        r"""
        Args:
            d (int): Dimension. 
            scale (Union[np.ndarray,torch.Tensor]): Scaling factor $S$.
            lengthscales (Union[np.ndarray,torch.Tensor]): Product weights $(\gamma_1,\dots,\gamma_d)$.
            alpha (Union[np.ndarray,torch.Tensor]): Smoothness parameters $(\alpha_1,\dots,\alpha_d)$ where $\alpha_j \geq 1$ for $j=1,\dots,d$.
            shape_scale (list): Shape of `scale` when `np.isscalar(scale)`. 
            shape_lengthscales (list): Shape of `lengthscales` when `np.isscalar(lengthscales)`
            tfs_scale (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            tfs_lengthscales (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            torchify (bool): If `True`, use the `torch` backend. Set to `True` if computing gradients with respect to inputs and/or hyperparameters.
            requires_grad_scale (bool): If `True` and `torchify`, set `requires_grad=True` for `scale`.
            requires_grad_lengthscales (bool): If `True` and `torchify`, set `requires_grad=True` for `lengthscales`.
            device (torch.device): If `torchify`, put things onto this device.
            compile_call (bool): If `True`, `torch.compile` the `parsed___call__` method. 
            comiple_call_kwargs (dict): When `compile_call` is `True`, pass these keyword arguments to `torch.compile`.
        """
        super().__init__(
            d = d, 
            scale = scale, 
            lengthscales = lengthscales,
            shape_scale = shape_scale,
            shape_lengthscales = shape_lengthscales, 
            tfs_scale = tfs_scale,
            tfs_lengthscales = tfs_lengthscales,
            torchify = torchify, 
            requires_grad_scale = requires_grad_scale, 
            requires_grad_lengthscales = requires_grad_lengthscales, 
            device = device,
            compile_call = compile_call,
            comiple_call_kwargs = comiple_call_kwargs,
        )
        self.raw_alpha,self.tf_alpha = self.parse_assign_param(
            pname = "alpha",
            param = alpha, 
            shape_param = [self.d],
            requires_grad_param = False,
            tfs_param = (tf_identity,tf_identity),
            endsize_ops = [self.d],
            constraints = ["POSITIVE"])
        assert self.alpha.shape==(self.d,)
        assert all(int(alphaj) in BERNOULLIPOLYSDICT for alphaj in self.alpha)
        if self.torchify:
            import torch 
            self.lgamma = torch.lgamma 
        else:
            self.lgamma = scipy.special.loggamma
    
    @property
    def alpha(self):
        return self.tf_alpha(self.raw_alpha)
    
    def parsed_single_integral_01d(self, x, batch_params):
        return batch_params["scale"][...,0]
    
    def double_integral_01d(self):
        return self.scale[...,0]
    
    def get_per_dim_components(self, x0, x1, beta0, beta1):
        p = len(beta0)
        betasum = beta0+beta1
        order = 2*self.alpha-betasum
        assert (2<=order).all(), "order must all be at least 2, but got order = %s"%str(order)
        coeffs = (-1)**(self.alpha+beta1+1)*self.npt.exp(2*self.alpha*np.log(2*np.pi)-self.lgamma(order+1))
        delta = (x0-x1)%1
        kperdim = coeffs*self.npt.stack([self.npt.concatenate([bernoulli_poly(int(order[l,j].item()),delta[...,j,None]) for j in range(self.d)],-1) for l in range(p)],-2)
        return kperdim
        
    def combine_per_dim_components(self, kperdim, beta0, beta1, c, batch_params):
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        ind = 1.*((beta0+beta1)==0)
        k = scale*((ind+lengthscales*kperdim).prod(-1)*c).sum(-1)
        return k
    
    def parsed___call__(self, x0, x1, beta0, beta1, c, batch_params):
        kperdim = self.get_per_dim_components(x0,x1,beta0,beta1)
        k = self.combine_per_dim_components(kperdim,beta0,beta1,c,batch_params)
        return k
