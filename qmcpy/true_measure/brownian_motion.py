from .gaussian import Gaussian
from ..discrete_distribution import Sobol
from ._true_measure import TrueMeasure
from ..util import ParameterError
from numpy import *


class BrownianMotion(Gaussian):
    """
    Geometric Brownian Motion.
    
    >>> d = 4
    >>> s = Sobol(d,seed=7)
    >>> bm = BrownianMotion(d)
    >>> bm
    BrownianMotion (TrueMeasure Object)
        d               2^(2)
        time_vec        [0.25 0.5  0.75 1.  ]
        drift           0
        mean            [0. 0. 0. 0.]
        covariance      [[0.25 0.25 0.25 0.25]
                        [0.25 0.5  0.5  0.5 ]
                        [0.25 0.5  0.75 0.75]
                        [0.25 0.5  0.75 1.  ]]
        decomp_type     pca
    >>> bm.transform(s.gen_samples(n_min=4,n_max=8))
    array([[ 0.664,  0.089, -0.049, -0.447],
           [-0.251,  0.358,  0.539,  0.835],
           [-0.691, -0.414, -0.463, -0.923],
           [ 0.882,  1.383,  1.689,  1.81 ]])
    >>> bm.set_transform(BrownianMotion(d,drift=4))
    >>> bm.transformer
    BrownianMotion (TrueMeasure Object)
        d               2^(2)
        time_vec        [0.25 0.5  0.75 1.  ]
        drift           2^(2)
        mean            [1. 2. 3. 4.]
        covariance      [[0.25 0.25 0.25 0.25]
                        [0.25 0.5  0.5  0.5 ]
                        [0.25 0.5  0.75 0.75]
                        [0.25 0.5  0.75 1.  ]]
        decomp_type     pca
    >>> bm.transformer.transform(s.gen_samples(n_min=4,n_max=8))
    array([[1.664, 2.089, 2.951, 3.553],
           [0.749, 2.358, 3.539, 4.835],
           [0.309, 1.586, 2.537, 3.077],
           [1.882, 3.383, 4.689, 5.81 ]])
    >>> d_new = 6
    >>> s.set_dimension(d_new)
    >>> bm.set_dimension(d_new)
    >>> bm
    BrownianMotion (TrueMeasure Object)
        d               6
        time_vec        [0.167 0.333 0.5   0.667 0.833 1.   ]
        drift           0
        mean            [0. 0. 0. 0. 0. 0.]
        covariance      [[0.167 0.167 0.167 0.167 0.167 0.167]
                        [0.167 0.333 0.333 0.333 0.333 0.333]
                        [0.167 0.333 0.5   0.5   0.5   0.5  ]
                        [0.167 0.333 0.5   0.667 0.667 0.667]
                        [0.167 0.333 0.5   0.667 0.833 0.833]
                        [0.167 0.333 0.5   0.667 0.833 1.   ]]
        decomp_type     pca
    >>> bm.transformer
    BrownianMotion (TrueMeasure Object)
        d               6
        time_vec        [0.167 0.333 0.5   0.667 0.833 1.   ]
        drift           2^(2)
        mean            [0.667 1.333 2.    2.667 3.333 4.   ]
        covariance      [[0.167 0.167 0.167 0.167 0.167 0.167]
                        [0.167 0.333 0.333 0.333 0.333 0.333]
                        [0.167 0.333 0.5   0.5   0.5   0.5  ]
                        [0.167 0.333 0.5   0.667 0.667 0.667]
                        [0.167 0.333 0.5   0.667 0.833 0.833]
                        [0.167 0.333 0.5   0.667 0.833 1.   ]]
        decomp_type     pca
    >>> bm.transformer.transform(s.gen_samples(n_min=4,n_max=8))
    array([[1.071, 2.026, 1.951, 2.591, 3.268, 3.453],
           [0.536, 1.135, 2.45 , 3.223, 3.863, 4.905],
           [0.057, 0.794, 1.529, 2.324, 2.639, 3.071],
           [1.383, 2.287, 3.497, 4.2  , 5.086, 5.777]])
    """

    parameters = ['d', 'time_vec', 'drift', 'mean', 'covariance', 'decomp_type']

    def __init__(self, dimension=1, t_final=1, drift=0):
        """
        Args:
            dimension (int): dimension of the distribution
            t_final (float): end time for the Brownian Motion. 
            drift (int): Gaussian mean is time_vec*drift
        """
        self.d = dimension
        self.t = t_final # exercise time
        self.drift = drift
        self.time_vec = linspace(self.t/self.d,self.t,self.d) # evenly spaced
        self.sigma_bm = array([[min(self.time_vec[i],self.time_vec[j]) for i in range(self.d)] for j in range(self.d)])
        self.drift_time_vec = self.drift*self.time_vec # mean
        super().__init__(dimension=self.d, mean=self.drift_time_vec, covariance=self.sigma_bm, decomp_type='PCA')

    def set_dimension(self, dimension):
        self.d = dimension
        self.time_vec = linspace(self.t/self.d,self.t,self.d) # evenly spaced
        self.sigma_bm = array([[min(self.time_vec[i],self.time_vec[j]) for i in range(self.d)] for j in range(self.d)])
        self.drift_time_vec = self.drift*self.time_vec # mean
        self._set_mean_cov(self.drift_time_vec,self.sigma_bm)
        if self.transformer!=self:
            self.transformer.set_dimension(self.d)
            

