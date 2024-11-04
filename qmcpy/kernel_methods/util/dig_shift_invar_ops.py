import numpy as np 

def k4sumterm(x, t):
    r""" 
    $$\sum_{a=0}**{t-1} x_a / 2**{3a}$$
    where $x_a$ is the bit at index $a$ in the binary expansion of $x$
    e.g. $x = 6$ with $t=3$ has $(x_0,x_1,x_2) = (1,1,0)

    Args:
        x (np.ndarray or torch.Tensor): integer arrays
        t (int): number of bits in each integer
    
    >>> t = 3
    >>> x = np.random.randint(0,2**t,(5,4))
    >>> k4sumterm(x,t).shape
    (5, 4)
    >>> import torch 
    >>> xt = torch.randint(0,2**t,(5,4))
    >>> k4sumterm(xt,t).shape
    torch.Size([5, 4])
    """
    total = 0.
    for a in range(0,t):
        total += (-1)**((x>>(t-a-1))&1)/float(2**(3*a))
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
    and $\mu_\alpha$ is the Dick weight function which sums the first $\alpha$ largest indices of 1 bits in the binary expansion of $k$ 
    e.g. $k=13=1101_2$ 1-bit indexes (4,3,1)$ so $\mu_1(k) = 4, \mu_2(k) = 4+3, \mu_3(k) = 4+3+1 = \mu_4(k) = \mu_5(k) = \dots$

    References: 
        
        [1] Dick, Josef. "Walsh spaces containing smooth functions and quasi–Monte Carlo rules of arbitrary high order." SIAM Journal on Numerical Analysis 46.3 (2008): 1519-1553.

        [2] Dick, Josef. "The decay of the Walsh coefficients of smooth functions." Bulletin of the Australian Mathematical Society 80.3 (2009): 430-453.

    Args:
        alpha (int): weighted walsh functions order 
        xb (np.ndarray or torch.Tensor): integer points at which to evaluate the weighted Walsh function 
        t (int): number of bits in each integer in xb
    
    >>> import torch 
    >>> t = 3 
    >>> xb = np.random.randint(0,2**t,(4,5))
    >>> xbt = torch.from_numpy(xb)
    >>> for alpha in list(WEIGHTEDWALSHFUNCSPOS.keys()):
    ...     y = weighted_walsh_funcs(alpha,xb,t)
    ...     yt = weighted_walsh_funcs(alpha,xbt,t)
    """
    assert isinstance(alpha,int)
    assert alpha in WEIGHTEDWALSHFUNCSPOS, "alpha = %d not in WEIGHTEDWALSHFUNCSPOS"%alpha
    assert alpha in WEIGHTEDWALSHFUNCSZEROS, "alpha = %d not in WEIGHTEDWALSHFUNCSZEROS"%alpha
    if isinstance(xb,np.ndarray):
        np_or_torch = np 
    else:
        import torch 
        np_or_torch = torch 
    y = np_or_torch.ones(xb.shape)
    pidxs = xb>0
    y[~pidxs] = WEIGHTEDWALSHFUNCSZEROS[alpha]
    xfpidxs = (2**(-t))*xb[pidxs]
    betapidxs = -np_or_torch.floor(np_or_torch.log2(xfpidxs))
    y[pidxs] = WEIGHTEDWALSHFUNCSPOS[alpha](betapidxs,xfpidxs,xb[pidxs],t)
    return y
    