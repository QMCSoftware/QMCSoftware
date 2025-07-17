from .abstract_kernel import AbstractKernelScaleLengthscales
from .util import tf_identity,tf_exp_eps,tf_exp_eps_inv,get_npt
from ...util import ParameterError
import numpy as np 
from typing import Union

def k4sumterm(x, t, cutoff=1e-8):
    r""" 
    $$K_4(x) = \sum_{a=0}^{t-1} \frac{x_a}{2^{3a}}$$

    where $x_a$ is the bit at index $a$ in the binary expansion of $x$
    e.g. $x = 6$ with $t=3$ has $(x_0,x_1,x_2) = (1,1,0)$

    Examples:
        >>> t = 3
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> x = rng.integers(low=0,high=2**t,size=(5,4))
        >>> with np.printoptions(precision=2):
        ...     k4sumterm(x,t)
        array([[ 1.11,  0.89,  1.11, -0.89],
               [ 1.11,  1.14,  1.11, -0.86],
               [-0.89,  0.86, -1.11,  0.89],
               [-1.11,  0.89, -0.89,  0.89],
               [-1.14, -0.89, -0.89, -0.86]])
        >>> import torch 
        >>> with torch._tensor_str.printoptions(precision=2):
        ...     k4sumterm(torch.from_numpy(x),t)
        tensor([[ 1.11,  0.89,  1.11, -0.89],
                [ 1.11,  1.14,  1.11, -0.86],
                [-0.89,  0.86, -1.11,  0.89],
                [-1.11,  0.89, -0.89,  0.89],
                [-1.14, -0.89, -0.89, -0.86]])
    
    Args:
        x (Union[np.ndarray torch.Tensor]): Integer arrays.
        t (int): Number of bits in each integer.
    
    Returns:
        y (Union[np.ndarray torch.Tensor]): The $K_4$ sum term.
    """
    total = 0.
    for a in range(0,t):
        factor = 1/float(2.**(3*a))
        if factor<cutoff: break
        total += (-1.)**((x>>(t-a-1))&1)*factor
    return total

WEIGHTEDWALSHFUNCSPOS = {
    2: lambda beta,xf,xb,t: -beta*xf + 5/2*(1-2**(-beta)),
    3: lambda beta,xf,xb,t: beta*xf**2 - 5*(1-2**(-beta))*xf + 43/18*(1-2**(-2*beta)),
    4: lambda beta,xf,xb,t: -2/3*beta*xf**3 + 5*(1-2**(-beta))*xf**2 - 43/9*(1-2**(-2*beta))*xf + 701/294*(1-2**(-3*beta)) + beta*(1/48*k4sumterm(xb,t)-1/42),
}

WEIGHTEDWALSHFUNCSZEROS = {
    2: 5/2,
    3: 43/18,
    4: 701/294,
}

def weighted_walsh_funcs(alpha, xb, t):
    r"""
    Weighted walsh functions 

    $$\sum_{k=0}^\infty \mathrm{wal}_k(x) 2^{-\mu_\alpha(k)}$$ 

    where $\mathrm{wal}_k$ is the $k^\text{th}$ Walsh function 
    and $\mu_\alpha$ is the Dick weight function which sums the first $\alpha$ largest indices of $1$ bits in the binary expansion of $k$ 
    e.g. $k=13=1101_2$ has 1-bit indexes $(4,3,1)$ so 
    
    $$\mu_1(k) = 4, \mu_2(k) = 4+3, \mu_3(k) = 4+3+1 = \mu_4(k) = \mu_5(k) = \dots$$

    Examples:
        >>> t = 3 
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> xb = rng.integers(low=0,high=2**t,size=(2,3))
        >>> available_alpha = list(WEIGHTEDWALSHFUNCSPOS.keys())
        >>> available_alpha
        [2, 3, 4]
        >>> for alpha in available_alpha:
        ...     y = weighted_walsh_funcs(alpha,xb,t)
        ...     with np.printoptions(precision=2):
        ...         print("alpha = %d\n%s"%(alpha,y))
        alpha = 2
        [[1.81 1.38 1.81]
         [0.62 1.81 2.5 ]]
        alpha = 3
        [[1.85 1.43 1.85]
         [0.62 1.85 2.39]]
        alpha = 4
        [[1.85 1.43 1.85]
         [0.62 1.85 2.38]]
        >>> import torch
        >>> for alpha in available_alpha:
        ...     y = weighted_walsh_funcs(alpha,torch.from_numpy(xb),t)
        ...     with torch._tensor_str.printoptions(precision=2):
        ...         print("alpha = %d\n%s"%(alpha,y))
        alpha = 2
        tensor([[1.81, 1.38, 1.81],
                [0.62, 1.81, 2.50]])
        alpha = 3
        tensor([[1.85, 1.43, 1.85],
                [0.62, 1.85, 2.39]])
        alpha = 4
        tensor([[1.85, 1.43, 1.85],
                [0.62, 1.85, 2.38]])
    
    Args:
        alpha (int): Weighted walsh functions order.
        xb (Union[np.ndarray,torch.Tensor]): Jnteger points at which to evaluate the weighted Walsh function.
        t (int): Number of bits in each integer in xb.
    
    returns:
        y (Union[np.ndarray,torch.Tensor]): Weighted Walsh function values.
    
    **References:**
        
    1.  Dick, Josef.  
        "Walsh spaces containing smooth functions and quasi–Monte Carlo rules of arbitrary high order."  
        SIAM Journal on Numerical Analysis 46.3 (2008): 1519-1553.

    2.  Dick, Josef.  
        "The decay of the Walsh coefficients of smooth functions."  
        Bulletin of the Australian Mathematical Society 80.3 (2009): 430-453.    
    """
    assert isinstance(alpha,int)
    assert alpha in WEIGHTEDWALSHFUNCSPOS, "alpha = %d not in WEIGHTEDWALSHFUNCSPOS"%alpha
    assert alpha in WEIGHTEDWALSHFUNCSZEROS, "alpha = %d not in WEIGHTEDWALSHFUNCSZEROS"%alpha
    if isinstance(xb,np.ndarray):
        np_or_torch = np 
        y = np.ones(xb.shape) 
    else:
        import torch 
        np_or_torch = torch 
        y = torch.ones(xb.shape,device=xb.device)
    pidxs = xb>0
    y[~pidxs] = WEIGHTEDWALSHFUNCSZEROS[alpha]
    xfpidxs = (2**(-t))*xb[pidxs]
    betapidxs = -np_or_torch.floor(np_or_torch.log2(xfpidxs))
    y[pidxs] = WEIGHTEDWALSHFUNCSPOS[alpha](betapidxs,xfpidxs,xb[pidxs],t)
    return y

def to_bin(x, t):
    r"""
    Convert floating point representations of digital net samples in base $b=2$ to binary representations.

    Examples:
        >>> xf = np.random.Generator(np.random.PCG64(7)).uniform(low=0,high=1,size=(5))
        >>> xf 
        array([0.62509547, 0.8972138 , 0.77568569, 0.22520719, 0.30016628])
        >>> xb = to_bin(xf,2)
        >>> xb
        array([2, 3, 3, 0, 1], dtype=uint64)
        >>> to_bin(xb,2) 
        array([2, 3, 3, 0, 1], dtype=uint64)
        >>> import torch 
        >>> xftorch = torch.from_numpy(xf) 
        >>> xftorch
        tensor([0.6251, 0.8972, 0.7757, 0.2252, 0.3002], dtype=torch.float64)
        >>> xbtorch = to_bin(xftorch,2)
        >>> xbtorch
        tensor([2, 3, 3, 0, 1])
        >>> to_bin(xbtorch,2)
        tensor([2, 3, 3, 0, 1])

    
    Args:
        x (Union[np.ndarray,torch.Tensor]): floating point representation of samples. 
        t (int): number of bits in binary represtnations. Typically `dnb2.t` where `isinstance(dnb2,DigitalNetB2)`.
    
    Returns: 
        xb (Unioin[np.ndarray,torch.Tensor]): binary representation of samples with `dtype` either `np.uint64` or `torch.int64`. 
    """
    npt = get_npt(x)
    if npt==np:
        if npt.issubdtype(x.dtype,npt.floating):
            return npt.floor((x%1)*2.**t).astype(npt.uint64)
        elif npt.issubdtype(x.dtype,npt.integer):
            return x
        else:
            raise ParameterError("x.dtype must be float or int, got %s"%str(x.dtype))
    else: # npt==torch
        if npt.is_floating_point(x):
            return npt.floor((x%1)*2.**t).to(npt.int64)
        elif (not npt.is_floating_point(x)) and (not npt.is_complex(x)): # int type 
            return x 
        else :
            raise ParameterError("x.dtype must be float or int, got %s"%str(x.dtype))
    return xb

def to_float(x, t):
    r"""
    Convert binary representations of digital net samples in base $b=2$ to floating point representations.
    
    Examples:
        >>> xb = np.arange(8,dtype=np.uint64)
        >>> xb 
        array([0, 1, 2, 3, 4, 5, 6, 7], dtype=uint64)
        >>> to_float(xb,3)
        array([0.   , 0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875])
        >>> xbtorch = bin_from_numpy_to_float(xb)
        >>> xbtorch
        tensor([0, 1, 2, 3, 4, 5, 6, 7])
        >>> to_float(xbtorch,3)
        tensor([0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750],
               dtype=torch.float64)
    
    Args:
        x (Union[np.ndarray,torch.Tensor]): binary representation of samples with `dtype` either `np.uint64` or `torch.int64`.
        t (int): number of bits in binary represtnations. Typically `dnb2.t` where `isinstance(dnb2,DigitalNetB2)`.
    
    Returns: 
        xf (Unioin[np.ndarray,torch.Tensor]): floating point representation of samples.  
    """
    npt = get_npt(x)
    if npt==np: # npt==torch
        if x.dtype==np.uint64: 
            return x.astype(np.float64)*2.**(-t)
        elif npt.is_floating_point(x):
            return x 
        else: 
            raise ParameterError("x.dtype must be np.uint64, got %s"%str(x.dtype))
    else:
        if x.dtype==npt.int64: 
            return x.to(npt.float64)*2.**(-t) 
        elif npt.is_floating_point(x):
            return x 
        else:
            raise ParameterError("x.dtype must be torch.int64, got %s"%str(x.dtype))

def bin_from_numpy_to_float(xb):
    r"""
    Convert `numpy.uint64` to `torch.int64`, useful for converting binary samples from `DigitalNetB2` to torch representations.
    
    Examples:
        >>> xb = np.arange(8,dtype=np.uint64)
        >>> xb 
        array([0, 1, 2, 3, 4, 5, 6, 7], dtype=uint64)
        >>> bin_from_numpy_to_float(xb)
        tensor([0, 1, 2, 3, 4, 5, 6, 7])
    
    Args:
        xb (Union[np.ndarray]): binary representation of samples with `dtype=np.uint64`
    
    Returns: 
        xbtorch (Unioin[torch.Tensor]): binary representation of samples with `dtype=torch.int64`.  
    """
    assert xb.dtype==np.uint64
    assert xb.max()<=(2**63-1), "require all xb < 2^63"
    import torch
    return torch.from_numpy(xb.astype(np.int64))

class KernelDigShiftInvar(AbstractKernelScaleLengthscales):
    
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
        >>> k00
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
        >>> xtorch = bin_from_numpy_to_float(x)
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

    Args:
        x (Union[np.ndarray,torch.Tensor]): First inputs with `x.shape=(*batch_shape_x,d)`.
        z (Union[np.ndarray,torch.Tensor]): Second inputs with shape `z.shape=(*batch_shape_z,d)`.
        t (int): number of bits in binary represtnations. Typically `dnb2.t` where `isinstance(dnb2,DigitalNetB2)`.
        alpha (Union[np.ndarray,torch.Tensor]): Smoothness parameters $(\alpha_1,\dots,\alpha_d)$ where $\alpha_j \geq 1$ for $j=1,\dots,d$.
        weights (Union[np.ndarray,torch.Tensor]): Product weights $(\gamma_1,\dots,\gamma_d)$ with shape `weights.shape=(d,)`.
        scale (float): Scaling factor $S$. 

    Returns: 
        k (Union[np.ndarray,torch.Tensor]): kernel values with `k.shape=(x+z).shape[:-1]`. 

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
    
    AUTOGRADKERNEL = False
    
    def __init__(self,
            d,
            t,
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
        self.t = t
        assert self.alpha.shape==(self.d,)
        assert all(1<=int(alphaj)<=4 for alphaj in self.alpha)
    
    @property
    def alpha(self):
        return self.tf_alpha(self.raw_alpha)
    
    def parsed_single_integral_01d(self, x):
        return self.scale[...,0] 
    
    def double_integral_01d(self):
        return self.scale[...,0]
    
    def get_per_dim_components(self, x0, x1, beta0, beta1):
        x0 = to_bin(x0,self.t) 
        x1 = to_bin(x1,self.t)
        betasum = beta0+beta1
        order = self.alpha-betasum
        assert (1<=order).all() and (order<=4).all(), "order must all be between 2 and 4, but got order = %s. Try increasing alpha"%str(order)
        assert not ((order==1)*(self.alpha>1)).any(), "taking the derivative of the order 2 digitally shift invariant kernel is not supported"
        ind = 1*(betasum>0)
        delta = x0^x1 
        kperdim = self.d*[None]
        for j in range(self.d):
            deltaj = delta[...,j,None]
            if order[j]==1: # order[j]=alpha[j] as we cannot take derivatives WRT the alpha=1 kernel and this cannot be the derivative of any kernels
                flog2deltaj = -self.npt.inf*self.npt.ones(deltaj.shape)
                pos = deltaj>0 
                flog2deltaj[pos] = self.npt.floor(self.npt.log2(deltaj[pos]))-self.t
                kperdim[j] = 6*(1/6-2**(flog2deltaj-1))
            else:
                kperdim[j] = weighted_walsh_funcs(int(self.alpha[j]),deltaj[...,None],self.t)[...,0]-1
        kperdim = (-2)**betasum*(ind+self.npt.concatenate(kperdim,-1))
        return kperdim
    
    def parsed___call__(self, x0, x1, beta0, beta1, batch_params):
        scale = batch_params["scale"][...,0]
        lengthscales = batch_params["lengthscales"]
        kperdim = self.get_per_dim_components(x0,x1,beta0,beta1)
        k = scale*(1+lengthscales*kperdim).prod(-1)
        return k
