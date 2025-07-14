import numpy as np 
from typing import Union
import scipy.special

class Polynomial():
    """
    >>> p = Polynomial([3,5,0,1,2])
    >>> x = np.random.rand(3,4)
    >>> y = p(x) 
    >>> y_true = 3*x**4 + 5*x**3 + 0*x**2 + 1*x**1 + 2*x**0
    >>> assert np.allclose(y,y_true,atol=1e-12)
    """
    def __init__(self, coeffs):
        """
        Polynomial evaluation with Horner's rule
        
        Args:
            coeffs (list or np.ndarray or torch.Tensor): vector of coefficients  
            e.g. coeffs = [a, b, c] corresponds to the quadratic polynomial a*x**2 + b*x + c
        """
        assert isinstance(coeffs,list)
        self.order = len(coeffs) 
        assert self.order >= 1
        self.coeffs = coeffs
    def __call__(self, x):
        if isinstance(x,np.ndarray):
            npt = np 
            powers = np.arange(self.order-1,-1,-1,dtype=x.dtype)
            coeffs = np.array(self.coeffs,dtype=x.dtype)
        else:
            import torch 
            npt = torch
            powers = torch.arange(self.order-1,-1,-1,dtype=x.dtype,device=x.device)
            coeffs = torch.tensor(self.coeffs,dtype=x.dtype,device=x.device)
        return (x[...,None]**powers*coeffs).sum(-1)

BERNOULLIPOLYSDICT = {
    #0:  Polynomial([1]),
    1:  Polynomial([1, -1/2]),
    2:  Polynomial([1, -1,   1/6]),
    3:  Polynomial([1, -3/2, 1/2,  0]),
    4:  Polynomial([1, -2,   1,    0, -1/30]),
    5:  Polynomial([1, -5/2, 5/3,  0, -1/6,  0]),
    6:  Polynomial([1, -3,   5/2,  0, -1/2,  0, 1/42]),
    7:  Polynomial([1, -7/2, 7/2,  0, -7/6,  0, 1/6, 0]),
    8:  Polynomial([1, -4,   14/3, 0, -7/3,  0, 2/3, 0, -1/30]),
    9:  Polynomial([1, -9/2, 6,    0, -21/5, 0, 2,   0, -3/10, 0]),
    10: Polynomial([1, -5,   15/2, 0, -7,    0, 5,   0, -3/2,  0, 5/66])
}

def bernoulli_poly(n, x):
    r"""
    $n^\text{th}$ Bernoulli polynomial

    Examples:
        >>> x = np.arange(6).reshape((2,3))/6
        >>> available_n = list(BERNOULLIPOLYSDICT.keys())
        >>> available_n
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> for n in available_n:
        ...     y = bernoulli_poly(n,x)
        ...     with np.printoptions(precision=2):
        ...         print("n = %d\n%s"%(n,y))
        n = 1
        [[-0.5  -0.33 -0.17]
         [ 0.    0.17  0.33]]
        n = 2
        [[ 0.17  0.03 -0.06]
         [-0.08 -0.06  0.03]]
        n = 3
        [[ 0.    0.05  0.04]
         [ 0.   -0.04 -0.05]]
        n = 4
        [[-0.03 -0.01  0.02]
         [ 0.03  0.02 -0.01]]
        n = 5
        [[ 0.00e+00 -2.19e-02 -2.06e-02]
         [ 1.39e-17  2.06e-02  2.19e-02]]
        n = 6
        [[ 0.02  0.01 -0.01]
         [-0.02 -0.01  0.01]]
        n = 7
        [[ 0.00e+00  2.28e-02  2.24e-02]
         [-1.39e-17 -2.24e-02 -2.28e-02]]
        n = 8
        [[-0.03 -0.02  0.02]
         [ 0.03  0.02 -0.02]]
        n = 9
        [[ 0.   -0.04 -0.04]
         [ 0.    0.04  0.04]]
        n = 10
        [[ 0.08  0.04 -0.04]
         [-0.08 -0.04  0.04]]
        >>> import scipy.special
        >>> for n in available_n:
        ...     bpoly_coeffs = BERNOULLIPOLYSDICT[n].coeffs
        ...     bpoly_coeffs_true = scipy.special.bernoulli(n)*scipy.special.comb(n,np.arange(n,-1,-1))
        ...     assert np.allclose(bpoly_coeffs_true,bpoly_coeffs,atol=1e-12)
    
    Args:
        n (int): Polynomial order.
        x (Union[np.ndarray,torch.Tensor]): Points at which to evaluate the Bernoulli polynomial.
    
    Returns:
        y (Union[np.ndarray,torch.Tensor]): Bernoulli polynomial values.
    """
    assert isinstance(n,int)
    assert n in BERNOULLIPOLYSDICT, "n = %d not in BERNOULLIPOLYSDICT"%n
    bpoly = BERNOULLIPOLYSDICT[n]
    y = bpoly(x) 
    return y

def kernel_shift_invar(x, z, alpha=1, weights=1, scale=1):
    r""" 
    Shift invariant kernel with 
    smoothness $\boldsymbol{\alpha}$, product weights $\boldsymbol{\gamma}$, and scale $S$:
    
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
        >>> weights = 1/np.arange(1,d+1)**2 
        >>> alpha = np.arange(1,d+1)
        >>> scale = 10
        >>> k00 = kernel_shift_invar(x[0],x[0],alpha=alpha,weights=weights,scale=scale)
        >>> k00.item()
        91.2344445339634
        >>> k0 = kernel_shift_invar(x,x[0],alpha=alpha,weights=weights,scale=scale)
        >>> with np.printoptions(precision=2):
        ...     print(k0)
        [91.23 -2.32  5.69  5.69 12.7  -4.78 -4.78 12.7 ]
        >>> assert k0[0]==k00
        >>> kmat = kernel_shift_invar(x[:,None,:],x[None,:,:],alpha=alpha,weights=weights,scale=scale)
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
        >>> kmat_torch = kernel_shift_invar(xtorch[:,None,:],xtorch[None,:,:],
        ...     alpha = torch.from_numpy(alpha),
        ...     weights = torch.from_numpy(weights),
        ...     scale = scale)
        >>> np.allclose(kmat_torch.numpy(),kmat)
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
    assert x.shape[-1]==z.shape[-1]
    d = x.shape[-1] 
    if isinstance(x,np.ndarray):
        npt = np
        lgamma = scipy.special.loggamma
    else:
        import torch 
        npt = torch
        lgamma = torch.lgamma 
        assert isinstance(x,torch.Tensor) and isinstance(z,torch.Tensor)
    alpha = npt.atleast_1d(alpha)*npt.ones(d,dtype=int)
    assert alpha.shape==(d,)
    assert (alpha%1==0).all() and (alpha>=1).all(), "alpha must all be positive ints"
    coeffs = (-1)**(alpha+1)*npt.exp((2*alpha)*np.log(2*np.pi)-lgamma(2*alpha+1))
    weights = npt.atleast_1d(weights)*npt.ones(d,dtype=int)
    assert weights.shape==(d,) and (weights>0).all()
    assert np.isscalar(scale) and scale>0
    k = scale
    for j in range(d):
        delta_j = (x[...,j,None]-z[...,j,None])%1
        ktilde = coeffs[j]*bernoulli_poly(int(2*alpha[j]),delta_j)[...,0]
        k = k*(1+weights[j]*ktilde)
    return k
