from .gaussian import Gaussian
from ..discrete_distribution.abstract_discrete_distribution import AbstractDiscreteDistribution
from ..discrete_distribution import DigitalNetB2
from .abstract_true_measure import AbstractTrueMeasure
from ..util import ParameterError, _univ_repr
import numpy as np
from typing import Union
import math
from scipy.stats import norm

class BrownianMotion(Gaussian):
    r"""
    Brownian Motion as described in [https://en.wikipedia.org/wiki/Brownian_motion](https://en.wikipedia.org/wiki/Brownian_motion).  
    For a standard Brownian Motion $W$ we define the Brownian Motion $B$ with initial value $B_0$, drift $\gamma$, and diffusion $\sigma^2$ to be

    $$B(t) = B_0 + \gamma t + \sigma W(t).$$
    
    Examples:
        >>> true_measure = BrownianMotion(DigitalNetB2(4,seed=7),t_final=2,drift=2)
        >>> true_measure(2)
        array([[0.82189263, 2.7851793 , 3.60126805, 3.98054724],
               [0.2610643 , 0.06170064, 1.06448269, 2.30990767]])
        >>> true_measure
        BrownianMotion (AbstractTrueMeasure)
            time_vec        [0.5 1.  1.5 2. ]
            drift           2^(1)
            mean            [1. 2. 3. 4.]
            covariance      [[0.5 0.5 0.5 0.5]
                             [0.5 1.  1.  1. ]
                             [0.5 1.  1.5 1.5]
                             [0.5 1.  1.5 2. ]]
            decomp_type     PCA
    
        With independent replications 

        >>> x = BrownianMotion(DigitalNetB2(3,seed=7,replications=2),t_final=2,drift=2)(4)
        >>> x.shape 
        (2, 4, 3)
        >>> x
        array([[[0.66154685, 1.50620966, 3.52322901],
                [1.77064217, 3.32782204, 4.45013223],
                [1.33558688, 3.26017547, 3.40692337],
                [2.10317345, 3.78961839, 6.17948096]],
        <BLANKLINE>
               [[1.77868019, 2.75347902, 3.41161419],
                [0.44891984, 2.53987304, 4.7224811 ],
                [0.23147948, 2.25289769, 3.00039101],
                [2.06762574, 3.21756319, 4.93375923]]])
    """

    def __init__(self, sampler, t_final=1, initial_value=0, drift=0, diffusion=1, decomp_type='PCA', bridge = "FALSE"):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): Either  
                
                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            t_final (float): End time. 
            initial_value (float): Initial value $B_0$. 
            drift (int): Drift $\gamma$. 
            diffusion (int): Diffusion $\sigma^2$. 
            decomp_type (str): Method for decomposition for covariance matrix. Options include
             
                - `'PCA'` for principal component analysis, or 
                - `'Cholesky'` for cholesky decomposition.
        """
        self.parameters = ['time_vec', 'drift', 'mean', 'covariance', 'decomp_type']
        # default to transform from standard uniform
        self.bridge = str(bridge).upper()
        assert self.bridge in ["TRUE", "FALSE"]
        self.domain = np.array([[0,1]])
        self._parse_sampler(sampler)
        self.t = t_final # exercise time
        self.initial_value = initial_value
        self.drift = drift
        self.diffusion = diffusion
        self.time_vec = np.linspace(self.t/self.d,self.t,self.d) # evenly spaced
        
        if self.bridge == "TRUE": 
            T =  self.t
            self.diffused_sigma_bm = self.diffusion*np.array
        else:
            self.diffused_sigma_bm = self.diffusion*np.array([[min(self.time_vec[i],self.time_vec[j]) for i in range(self.d)] for j in range(self.d)])

        self.drift_time_vec_plus_init = self.drift*self.time_vec+self.initial_value # mean
        self._parse_gaussian_params(self.drift_time_vec_plus_init,self.diffused_sigma_bm,decomp_type)
        self.range = np.array([[-np.inf,np.inf]])
        
        

        super(Gaussian,self).__init__()


    def _transform(self, x):
        z = norm.ppf(x)
        if self.bridge == "TRUE": 
            path = self.createPath_Bridge(z)
        else: 
            path = super._transform(x)
        return path

    def _createPath_Bridge(self, z):
        w_f = z[-1] * math.sqrt(self.t)
        w_all = [None]*(self.d+1)
        w_all[0] = 0
        w_all[-1] = w_f
        self._createPath_Bridge_helper(0, self.d, w_all, z)
        w_all = np.array(w_all)
        return w_all
    
    def _createPath_Bridge_helper(self, initial, final, w_all, z):
        if (final - initial <= 1):
            return; 
    
        mid = (final + initial)//2
        #calculate actual time value 
        t_f = self.time_vec[final]
        t_i = self.time_vec[initial]
        t = mid/self.dim * self.time

        #calculate value at mid index 
        mean = ((t_f - t)*w_all[initial] + (t - t_i)*w_all[final])/(t_f - t_i)
        std = math.sqrt((t_f - t)*(t - t_i)/(t_f - t_i)) * z[mid - 1]
        w_t = mean + std
        w_all[mid] = w_t
        
        self._createPath_Bridge_helper(mid, final, w_all, z)
        self._createPath_Bridge_helper(initial, mid, w_all, z)
    
    def _spawn(self, sampler, dimension):
        return BrownianMotion(sampler,t_final=self.t,drift=self.drift,decomp_type=self.decomp_type)
    
