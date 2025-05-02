from .gaussian import Gaussian
from ..discrete_distribution.abstract_discrete_distribution import AbstractDiscreteDistribution
from ..discrete_distribution import DigitalNetB2
from ._true_measure import TrueMeasure
from ..util import ParameterError, _univ_repr
import numpy as np


class BrownianMotion(Gaussian):
    """
    Geometric Brownian Motion.
    
    >>> bm = BrownianMotion(DigitalNetB2(4,seed=7),t_final=2,drift=2)
    >>> bm.gen_samples(2)
    array([[0.99572233, 2.32327545, 3.03550196, 2.48206084],
           [1.02615537, 2.22809234, 3.0408864 , 4.41879786]])
    >>> bm
    BrownianMotion (TrueMeasure Object)
        time_vec        [0.5 1.  1.5 2. ]
        drift           2^(1)
        mean            [1. 2. 3. 4.]
        covariance      [[0.5 0.5 0.5 0.5]
                        [0.5 1.  1.  1. ]
                        [0.5 1.  1.5 1.5]
                        [0.5 1.  1.5 2. ]]
        decomp_type     PCA
    """

    def __init__(self, sampler, t_final=1, initial_value=0, drift=0, diffusion=1, decomp_type='PCA'):
        r"""
        BrownianMotion(t) = (initial_value) + (drift)*t + \sqrt{diffusion}*StandardBrownianMotion(t)

        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            t_final (float): end time for the Brownian Motion. 
            initial_value (float): See above formula
            drift (int): See above formula
            diffusion (int): See above formula
            decomp_type (str): method of decomposition either  
                "PCA" for principal component analysis or 
                "Cholesky" for cholesky decomposition.
        """
        self.parameters = ['time_vec', 'drift', 'mean', 'covariance', 'decomp_type']
        # default to transform from standard uniform
        self.domain = np.array([[0,1]])
        self._parse_sampler(sampler)
        self.t = t_final # exercise time
        self.initial_value = initial_value
        self.drift = drift
        self.diffusion = diffusion
        self.time_vec = np.linspace(self.t/self.d,self.t,self.d) # evenly spaced
        self.diffused_sigma_bm = self.diffusion*np.array([[min(self.time_vec[i],self.time_vec[j]) for i in range(self.d)] for j in range(self.d)])
        self.drift_time_vec_plus_init = self.drift*self.time_vec+self.initial_value # mean
        self._parse_gaussian_params(self.drift_time_vec_plus_init,self.diffused_sigma_bm,decomp_type)
        self.range = np.array([[-np.inf,np.inf]])
        super(Gaussian,self).__init__()
    
    def _spawn(self, sampler, dimension):
        return BrownianMotion(sampler,t_final=self.t,drift=self.drift,decomp_type=self.decomp_type)
        
