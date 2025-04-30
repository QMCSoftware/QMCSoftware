import numpy as np 

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
            powers = torch.np.arange(self.order-1,-1,-1,dtype=x.dtype,device=x.device)
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
    $n^\text{th} Bernoulli polynomial

    Args:
        n (int): polynomial order
        x (np.ndarray or torch.Tensor): points at which to evaluate the Bernoulli polynomial
    
    >>> import scipy.special
    >>> import numpy as np
    >>> x = np.arange(8).reshape(4,2)/8
    >>> for n,bpoly in BERNOULLIPOLYSDICT.items():
    ...     bvec = scipy.special.bernoulli(n)
    ...     choosevec = scipy.special.comb(n,np.arange(n,-1,-1))
    ...     bpoly_coeffs_true = bvec*choosevec
    ...     assert np.allclose(bpoly_coeffs_true,bpoly.coeffs,atol=1e-12)
    ...     y = bernoulli_poly(n,x)
    """
    assert isinstance(n,int)
    assert n in BERNOULLIPOLYSDICT, "n = %d not in BERNOULLIPOLYSDICT"%n
    bpoly = BERNOULLIPOLYSDICT[n]
    y = bpoly(x) 
    return y
