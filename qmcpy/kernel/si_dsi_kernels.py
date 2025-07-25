from .abstract_kernel import AbstractKernelScaleLengthscales
from ..util.transforms import tf_exp_eps,tf_exp_eps_inv,tf_identity
from ..util.shift_invar_ops import BERNOULLIPOLYSDICT,bernoulli_poly
from ..util.transforms import tf_identity,tf_exp_eps,tf_exp_eps_inv
from ..util.dig_shift_invar_ops import to_bin,weighted_walsh_funcs,bin_from_numpy_to_torch,to_float
from ..util import ParameterError
import numpy as np 
from typing import Union,Tuple
import scipy.special


class AbstractSIDSIKernel(AbstractKernelScaleLengthscales):

    AUTOGRADKERNEL = False

    def __init__(self,
            d, 
            scale, 
            lengthscales,
            alpha,
            shape_scale,
            shape_lengthscales, 
            tfs_scale,
            tfs_lengthscales,
            torchify, 
            requires_grad_scale, 
            requires_grad_lengthscales, 
            device,
            compile_call,
            comiple_call_kwargs,
            ):
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
    
    @property
    def alpha(self):
        return self.tf_alpha(self.raw_alpha)
    
    def parsed_single_integral_01d(self, x, batch_params):
        return batch_params["scale"][...,0]
    
    def double_integral_01d(self):
        return self.scale[...,0]
    
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
        

class KernelShiftInvar(AbstractSIDSIKernel):

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
            alpha = alpha, 
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
        assert all(int(alphaj) in BERNOULLIPOLYSDICT for alphaj in self.alpha)
        if self.torchify:
            import torch 
            self.lgamma = torch.lgamma 
        else:
            self.lgamma = scipy.special.loggamma
    
    def get_per_dim_components(self, x0, x1, beta0, beta1):
        p = len(beta0)
        betasum = beta0+beta1
        order = 2*self.alpha-betasum
        assert (2<=order).all(), "order must all be at least 2, but got order = %s"%str(order)
        coeffs = (-1)**(self.alpha+beta1+1)*self.npt.exp(2*self.alpha*np.log(2*np.pi)-self.lgamma(order+1))
        delta = (x0-x1)%1
        kperdim = coeffs*self.npt.stack([self.npt.concatenate([bernoulli_poly(int(order[l,j].item()),delta[...,j,None]) for j in range(self.d)],-1) for l in range(p)],-2)
        return kperdim


class KernelDigShiftInvar(AbstractSIDSIKernel):
    
    r""" 
    Digitally shift invariant kernel in base $b=2$ with 
    smoothness $\boldsymbol{\alpha}$, product weights $\boldsymbol{\gamma}$, and scale $S$: 
    
    $$\begin{aligned}
        K(\boldsymbol{x},\boldsymbol{z}) &= S \prod_{j=1}^d \left(1+ \gamma_j \tilde{K}_{\alpha_j}(x_j \oplus z_j)\right), \qquad\mathrm{where} \\
        \tilde{K}_1(x) &= \sum_{k \in \mathbb{N}} \frac{\mathrm{wal}_k(x)}{2^{2 \lfloor \log_2(x) \rfloor}} = 6 \left(\frac{1}{6} - 2^{\lfloor \log_2(x) \rfloor -1}\right), \\
        \tilde{K}_2(x) &= \sum_{k \in \mathbb{N}} \frac{\mathrm{wal}_k(x)}{2^{\mu_2(k)}} = -\beta(x) x + \frac{5}{2}\left[1-t_1(x)\right]-1, \\
        \tilde{K}_3(x) &= \sum_{k \in \mathbb{N}} \frac{\mathrm{wal}_k(x)}{2^{\mu_3(k)}} = \beta(x)x^2-5\left[1-t_1(x)\right]x+\frac{43}{18}\left[1-t_2(x)\right]-1, \\
        \tilde{K}_4(x) &= \sum_{k \in \mathbb{N}} \frac{\mathrm{wal}_k(x)}{2^{\mu_4(k)}} = - \frac{2}{3}\beta(x)x^3+5\left[1-t_1(x)\right]x^2 - \frac{43}{9}\left[1-t_2(x)\right]x +\frac{701}{294}\left[1-t_3(x)\right]+\beta(x)\left[\frac{1}{48}\sum_{a=0}^\infty \frac{\mathrm{wal}_{2^a}(x)}{2^{3a}} - \frac{1}{42}\right] - 1.
    \end{aligned}$$

    where 
    
    - $x \oplus z$ is XOR between bits, 
    - $\mathrm{wal}_k$ is the $k^\text{th}$ Walsh function, 
    - $\beta(x) = - \lfloor \log_2(x) \rfloor$ and $t_\nu(x) = 2^{-\nu \beta(x)}$ where $\beta(0)=t_\nu(0) = 0$, and 
    - and $\mu_\alpha$ is the Dick weight function which sums the first $\alpha$ largest indices of $1$ bits in the binary expansion of $k$ 
    e.g. $k=13=1101_2$ has 1-bit indexes $(4,3,1)$ so 
    
    $$\mu_1(k) = 4, \mu_2(k) = 4+3, \mu_3(k) = 4+3+1 = \mu_4(k) = \mu_5(k) = \dots.$$

    Examples:
        >>> from qmcpy import DigitalNetB2, fwht
        >>> n = 8
        >>> d = 4
        >>> dnb2 = DigitalNetB2(d,seed=11)
        >>> x = dnb2(n,return_binary=True)
        >>> x.shape
        (8, 4)
        >>> x.dtype
        dtype('uint64')
        >>> kernel = KernelDigShiftInvar(
        ...     d = d, 
        ...     t = dnb2.t,
        ...     alpha = list(range(1,d+1)),
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)])
        >>> k00 = kernel(x[0],x[0])
        >>> k00.item()
        34.490370029184525
        >>> k0 = kernel(x,x[0])
        >>> with np.printoptions(precision=2):
        ...     print(k0)
        [34.49  4.15  9.59  4.98 15.42  5.45 11.99  4.51]
        >>> assert k0[0]==k00
        >>> kmat = kernel(x[:,None,:],x[None,:,:])
        >>> with np.printoptions(precision=2):
        ...     print(kmat)
        [[34.49  4.15  9.59  4.98 15.42  5.45 11.99  4.51]
         [ 4.15 34.49  4.98  9.59  5.45 15.42  4.51 11.99]
         [ 9.59  4.98 34.49  4.15 11.99  4.51 15.42  5.45]
         [ 4.98  9.59  4.15 34.49  4.51 11.99  5.45 15.42]
         [15.42  5.45 11.99  4.51 34.49  4.15  9.59  4.98]
         [ 5.45 15.42  4.51 11.99  4.15 34.49  4.98  9.59]
         [11.99  4.51 15.42  5.45  9.59  4.98 34.49  4.15]
         [ 4.51 11.99  5.45 15.42  4.98  9.59  4.15 34.49]]
        >>> assert (kmat[:,0]==k0).all()
        >>> lam = np.sqrt(n)*fwht(k0)
        >>> y = np.random.Generator(np.random.PCG64(7)).uniform(low=0,high=1,size=(n))
        >>> np.allclose(fwht(fwht(y)*lam),kmat@y)
        True
        >>> np.allclose(fwht(fwht(y)/lam),np.linalg.solve(kmat,y))
        True
        >>> import torch 
        >>> xtorch = bin_from_numpy_to_torch(x)
        >>> kernel_torch = KernelDigShiftInvar(
        ...     d = d, 
        ...     t = dnb2.t,
        ...     alpha = list(range(1,d+1)),
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)],
        ...     torchify = True)
        >>> kmat_torch = kernel_torch(xtorch[:,None,:],xtorch[None,:,:])
        >>> np.allclose(kmat_torch.detach().numpy(),kmat)
        True
        >>> xf = to_float(x,dnb2.t)
        >>> kmat_from_floats = kernel(xf[:,None,:],xf[None,:,:])
        >>> np.allclose(kmat,kmat_from_floats)
        True
        >>> xftorch = to_float(xtorch,dnb2.t)
        >>> xftorch.dtype
        torch.float64
        >>> kmat_torch_from_floats = kernel_torch(xftorch[:,None,:],xftorch[None,:,:])
        >>> torch.allclose(kmat_torch_from_floats,kmat_torch)
        True
        >>> kernel.single_integral_01d(x)
        array([10.])
        >>> kernel_torch.single_integral_01d(xtorch)
        tensor([10.], grad_fn=<SelectBackward0>)

    **References:**
        
    1.  Dick, Josef.  
        "Walsh spaces containing smooth functions and quasi–Monte Carlo rules of arbitrary high order."  
        SIAM Journal on Numerical Analysis 46.3 (2008): 1519-1553.

    2.  Dick, Josef.  
        "The decay of the Walsh coefficients of smooth functions."  
        Bulletin of the Australian Mathematical Society 80.3 (2009): 430-453.  

    3.  Dick, Josef, and Friedrich Pillichshammer.  
        "Multivariate integration in weighted Hilbert spaces based on Walsh functions and weighted Sobolev spaces."  
        Journal of Complexity 21.2 (2005): 149-195.

    4.  Jagadeeswaran, Rathinavel, and Fred J. Hickernell.  
        "Fast automatic Bayesian cubature using Sobol’sampling."  
        Advances in Modeling and Simulation: Festschrift for Pierre L'Ecuyer. Cham: Springer International Publishing, 2022. 301-318.

    5.  Rathinavel, Jagadeeswaran.  
        Fast automatic Bayesian cubature using matching kernels and designs.  
        Illinois Institute of Technology, 2019.
    
    6.  Sorokin, Aleksei.  
        "A Unified Implementation of Quasi-Monte Carlo Generators, Randomization Routines, and Fast Kernel Methods."  
        arXiv preprint arXiv:2502.14256 (2025).
    """
        
    def __init__(self,
            d,
            t,
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
            t (int): number of bits in binary represtnations. Typically `dnb2.t` where `isinstance(dnb2,DigitalNetB2)`.
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
            alpha = alpha, 
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
        self.t = t
        assert all(1<=int(alphaj)<=4 for alphaj in self.alpha)
    
    def get_per_dim_components(self, x0, x1, beta0, beta1):
        x0 = to_bin(x0,self.t)
        x1 = to_bin(x1,self.t)
        p = len(beta0)
        betasum = beta0+beta1
        order = self.alpha-betasum
        assert (1<=order).all() and (order<=4).all(), "order must all be between 2 and 4, but got order = %s. Try increasing alpha"%str(order)
        assert not ((order==1)*(self.alpha>1)).any(), "taking the derivative of the order 2 digitally shift invariant kernel is not supported"
        ind = 1.*(betasum>0)
        delta = x0^x1 
        kparts = [None]*p
        for l in range(p):
            kparts_l = [None]*self.d
            for j in range(self.d):
                deltaj = delta[...,j,None]
                if order[l,j]==1: # order[j]=alpha[j] as we cannot take derivatives WRT the alpha=1 kernel and this cannot be the derivative of any kernels
                    flog2deltaj = -self.npt.inf*self.npt.ones(deltaj.shape)
                    pos = deltaj>0 
                    flog2deltaj[pos] = self.npt.floor(self.npt.log2(deltaj[pos]))-self.t
                    kparts_l[j] = 6*(1/6-2**(flog2deltaj-1))
                else:
                    kparts_l[j] = weighted_walsh_funcs(int(self.alpha[j]),deltaj[...,None],self.t)[...,0]-1
            kparts[l] = self.npt.concatenate(kparts_l,-1)
        kparts = self.npt.stack(kparts,-2)
        kperdim = (-2)**betasum*(ind+kparts)
        return kperdim