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
        factor = 1/float(2**(3*a))
        if factor<cutoff: break
        total += (-1)**((x>>(t-a-1))&1)*factor
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

    $$\sum_{k=0}^\infty \phi_k(x) 2^{-\mu_\alpha(k)}$$ 

    where $\phi_k$ is the $k^\text{th}$ Walsh function 
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
