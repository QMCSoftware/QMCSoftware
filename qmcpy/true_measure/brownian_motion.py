from .gaussian import Gaussian
from ..discrete_distribution import Sobol
from numpy import *


class BrownianMotion(Gaussian):
    """
    Geometric Brownian Motion.
    
    >>> d = 2
    >>> s = Sobol(d,seed=7)
    >>> bm = BrownianMotion(d,drift=1)
    >>> bm
    BrownianMotion (Gaussian Object)
        d               2
        time_vector     [0.5 1. ]
        drift           1
    >>> bm.transform(s.gen_samples(n_min=4,n_max=8))
    array([[ 0.04 ,  1.408],
           [ 0.413,  0.161],
           [ 1.083,  1.769],
           [-0.931, -0.879]])
    >>> d_new = 4
    >>> s.set_dimension(d)
    >>> bm.set_dimension(d)
    >>> bm
    BrownianMotion (Gaussian Object)
        time_vector     [0.25 0.5  0.75 1.  ]
        drift           1
    """

    parameters = ['time_vector','drift',]

    def __init__(self, dimension=1, t_final=1):
        """
        Args:
            dimension (int): dimension of the distribution
            t_final (float): end time for the Brownian Motion. 
        """
        self.d = dimension
        self.t = t_final # exercise time
        self.time_vector = linspace(self.t/self.d,self.t,self.d) # evenly spaced
        self.sigma = array([[min(self.time_vector[i],self.time_vector[j]) for i in range(self.d)] for j in range(self.d)])
        super().__init__(dimension=self.d, mean=zeros(self.d), covariance=self.sigma, decomp_type='PCA')