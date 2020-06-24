from ._true_measure import TrueMeasure
from ..discrete_distribution import Sobol
from ..util import TransformError, ParameterError
from numpy import linspace, cumsum, diff, insert, sqrt, array, exp, array, dot
from scipy.stats import norm
from scipy.linalg import cholesky


class BrownianMotion(TrueMeasure):
    """
    Geometric Brownian Motion.
    
    >>> dd = Sobol(2,seed=7)
    >>> bm = BrownianMotion(dd,drift=1)
    >>> bm
    BrownianMotion (TrueMeasure Object)
        distrib_name    Sobol
        time_vector     [ 0.500  1.000]
        drift           1
    >>> bm.gen_mimic_samples(n_min=4,n_max=8)
    array([[ 1.254,  1.296],
           [ 0.241,  1.237],
           [ 0.692,  1.207],
           [-0.379, -1.567]])
    >>> bm.set_dimension(4)
    >>> bm
    BrownianMotion (TrueMeasure Object)
        distrib_name    Sobol
        time_vector     [ 0.250  0.500  0.750  1.000]
        drift           1
    >>> bm.gen_mimic_samples(n_min=2,n_max=4)
    array([[ 0.559,  0.254,  0.632,  0.696],
           [-0.116,  0.304, -0.084,  0.694]])
    """

    parameters = ['time_vector','drift']

    def __init__(self, distribution, drift=0.):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
            drift (float): mean shift for importance sampling. 
        """
        self.distribution = distribution
        self.drift = float(drift)
        self.d = self.distribution.dimension
        self.time_vector = linspace(1./self.d,1,self.d) # evenly spaced
        self.ms_vec = self.drift * self.time_vector
        sigma = array([[min(self.time_vector[i],self.time_vector[j])
                        for i in range(self.d)]
                        for j in range(self.d)])
        self.a = cholesky(sigma).T
        self.t = 1.
        super(BrownianMotion,self).__init__()
    
    def _tf_to_mimic_samples(self, samples):
        """
        Transform samples to appear BrownianMotion.
        
        Args:
            samples (ndarray): samples from a discrete distribution
        
        Return:
            ndarray: samples from the DiscreteDistribution transformed to mimic the Brownain Motion.
        """
        if self.distribution.mimics == 'StdGaussian':
            # insert start time then cumulative sum over monitoring times
            std_gaussian_samples = samples
        elif self.distribution.mimics == "StdUniform":
            # inverse CDF, insert start time, then cumulative sum over monitoring times
            std_gaussian_samples = norm.ppf(samples)
        else:
            raise TransformError(\
                'Cannot transform samples mimicing %s to Brownian Motion'%self.distribution.mimics)
        mimic_samples = dot(self.a,std_gaussian_samples.T).T + self.ms_vec
        return mimic_samples

    def transform_g_to_f(self, g):
        """ See abstract method. """
        def f(samples, *args, **kwargs):
            z = self._tf_to_mimic_samples(samples)
            y = g(z,*args,**kwargs) * exp( (self.drift*self.t/2. - z[:,-1]) * self.drift)
            return y
        return f
    
    def gen_mimic_samples(self, *args, **kwargs):
        """ See abstract method. """
        samples = self.distribution.gen_samples(*args,**kwargs)
        mimic_samples = self._tf_to_mimic_samples(samples)
        return mimic_samples
    
    def set_dimension(self, dimension):
        """
        See abstract method. 
        
        Note:
            Monitoring times are evenly spaced as linspace(1/dimension,1,dimension)
        """
        self.distribution.set_dimension(dimension)
        self.d = dimension
        self.time_vector = linspace(1./self.d,1,self.d)
        self.ms_vec = self.drift * self.time_vector
        sigma = array([[min(self.time_vector[i],self.time_vector[j])
                        for i in range(self.d)]
                        for j in range(self.d)])
        self.a = cholesky(sigma).T
