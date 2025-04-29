from .gaussian import Gaussian
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..discrete_distribution import DigitalNetB2
from ._true_measure import TrueMeasure
from ..util import ParameterError, _univ_repr
from numpy import *


class BrownianMotion(Gaussian):
    """
    Geometric Brownian Motion.
    
    >>> bm = BrownianMotion(DigitalNetB2(4,seed=7),t_final=2,drift=2)
    >>> bm.drift_time_vec_plus_init
    array([1., 2., 3., 4.])
    >>> bm.diffused_sigma_bm
    array([[0.5, 0.5, 0.5, 0.5],
           [0.5, 1. , 1. , 1. ],
           [0.5, 1. , 1.5, 1.5],
           [0.5, 1. , 1.5, 2. ]])
    >>> from numpy.linalg import eigh 
    >>> evals,evecs = eigh(bm.diffused_sigma_bm)
    >>> evals 
    array([0.14155929, 0.21301102, 0.5       , 4.14542968])
    >>> evecs
    array([[-4.28525073e-01,  6.56538502e-01,  5.77350269e-01,
            -2.28013429e-01],
           [ 6.56538502e-01, -2.28013429e-01,  5.77350269e-01,
            -4.28525073e-01],
           [-5.77350269e-01, -5.77350269e-01, -1.96016445e-17,
            -5.77350269e-01],
           [ 2.28013429e-01,  4.28525073e-01, -5.77350269e-01,
            -6.56538502e-01]])
    >>> bm.gen_samples(2)
    array([[1.37811552, 3.04193956, 4.0037555 , 3.58311815],
           [0.84270956, 1.88332698, 2.5763853 , 3.89058669]])
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
        self.domain = array([[0,1]])
        self._parse_sampler(sampler)
        self.t = t_final # exercise time
        self.initial_value = initial_value
        self.drift = drift
        self.diffusion = diffusion
        self.time_vec = linspace(self.t/self.d,self.t,self.d) # evenly spaced
        self.diffused_sigma_bm = self.diffusion*array([[minimum(self.time_vec[i],self.time_vec[j]) for i in range(self.d)] for j in range(self.d)])
        self.drift_time_vec_plus_init = self.drift*self.time_vec+self.initial_value # mean
        self._parse_gaussian_params(self.drift_time_vec_plus_init,self.diffused_sigma_bm,decomp_type)
        self.range = array([[-inf,inf]])
        super(Gaussian,self).__init__()
    
    def _spawn(self, sampler, dimension):
        return BrownianMotion(sampler,t_final=self.t,drift=self.drift,decomp_type=self.decomp_type)
        
