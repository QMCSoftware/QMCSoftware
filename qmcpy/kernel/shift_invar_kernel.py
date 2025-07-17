from .abstract_kernel import AbstractKernelScaleLengthscales
from ..util.transforms import tf_exp_eps,tf_exp_eps_inv,tf_identity
from ..util.shift_invar_ops import BERNOULLIPOLYSDICT,bernoulli_poly
from ..util import ParameterError
import numpy as np 
from typing import Union
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
        
    **References:** 
    
    1.  Kaarnioja, Vesa, Frances Y. Kuo, and Ian H. Sloan.  
        "Lattice-based kernel approximation and serendipitous weights for parametric PDEs in very high dimensions."  
        International Conference on Monte Carlo and Quasi-Monte Carlo Methods in Scientific Computing. Cham: Springer International Publishing, 2022.
    
    Args:
        x (Union[np.ndarray,torch.Tensor]): First inputs with `x.shape=(*batch_shape_x,d)`.
        z (Union[np.ndarray,torch.Tensor]): Second inputs with shape `z.shape=(*batch_shape_z,d)`.
        alpha (Union[np.ndarray,torch.Tensor]): Smoothness parameters $(\alpha_1,\dots,\alpha_d)$ where $\alpha_j \geq 1$ for $j=1,\dots,d$.
        weights (Union[np.ndarray,torch.Tensor]): Product weights $(\gamma_1,\dots,\gamma_d)$ with shape `weights.shape=(d,)`.
        scale (float): Scaling factor $S$. 

    Returns: 
        k (Union[np.ndarray,torch.Tensor]): kernel values with `k.shape=(x+z).shape[:-1]`. 
    """
    
    AUTOGRADKERNEL = False
    
    def __init__(self,
            d, 
            scale = 1., 
            lengthscales = 1.,
            alpha = 2,
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
    
    def parsed_single_integral_01d(self, x):
        return self.scale[...,0] 
    
    def double_integral_01d(self):
        return self.scale[...,0]
    
    def get_per_dim_components(self, x0, x1, beta0, beta1):
        betasum = beta0+beta1
        order = 2*self.alpha-betasum
        assert (2<=order).all(), "order must all be at least 2, but got order = %s"%str(order)
        coeffs = (-1)**(self.alpha+beta1+1)*self.npt.exp(2*self.alpha*np.log(2*np.pi)-self.lgamma(order+1))
        delta = (x0-x1)%1
        kperdim = coeffs*self.npt.concatenate([bernoulli_poly(int(order[j].item()),delta[...,j,None]) for j in range(self.d)],-1)
        return kperdim
        
    def parsed___call__(self, x0, x1, beta0, beta1, batch_params):
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        kperdim = self.get_per_dim_components(x0,x1,beta0,beta1)
        k = scale*(1+lengthscales*kperdim).prod(-1)
        return k
