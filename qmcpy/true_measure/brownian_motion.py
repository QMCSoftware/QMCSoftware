""" Definition of BrownianMotion, a concrete implementation of TrueMeasure """

from ._true_measure import TrueMeasure
from ..util import TransformError, ParameterError
from numpy import linspace, cumsum, diff, insert, sqrt, array, exp, array, dot
from scipy.stats import norm
from scipy.linalg import cholesky


class BrownianMotion(TrueMeasure):
    """ Brownian Motion TrueMeasure """

    parameters = ['time_vector']

    def __init__(self, distribution, mean_shift_is=0):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
            time_vector (list of ndarrays): monitoring times for the Integrand's'
            mean_shift_is (float): mean shift for importance sampling. 
        """
        self.distribution = distribution
        self.mean_shift_is = mean_shift_is
        self.d = self.distribution.dimension
        self.time_vector = linspace(1./self.d,1,self.d)
        self.ms_vec = self.mean_shift_is * self.time_vector
        sigma = array([[min(self.time_vector[i],self.time_vector[j])
                        for i in range(self.d)]
                        for j in range(self.d)])
        self.a = cholesky(sigma).T
        self.t = 1
        super().__init__()
    
    def _tf_to_mimic_samples(self, samples):
        """
        Transform samples to appear BrownianMotion
        
        Args:
            samples (ndarray): samples from a discrete distribution
        
        Return:
             mimic_samples (ndarray): samples from the DiscreteDistribution transformed to appear 
                                  to appear like the TrueMeasure object
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
        """
        Transform g, the origianl integrand, to f,
        the integrand accepting standard distribution samples.  
        
        Args:
            g (method): original integrand
        
        Returns:
            f (method): transformed integrand
        """
        def f(samples, *args, **kwargs):
            z = self._tf_to_mimic_samples(samples)
            y = g(z,*args,**kwargs) * exp( (self.mean_shift_is*self.t/2 - z[:,-1]) * self.mean_shift_is)
            return y
        return f
    
    def gen_mimic_samples(self, *args, **kwargs):
        """
        Generate samples from the DiscreteDistribution object
        and transform them to mimic TrueMeasure samples
        
        Args:
            *args (tuple): Ordered arguments to self.distribution.gen_samples
            **kwrags (dict): Keyword arguments to self.distribution.gen_samples
        
        Return:
             mimic_samples (ndarray): samples from the DiscreteDistribution transformed to appear 
                                  to appear like the TrueMeasure object
        """
        samples = self.distribution.gen_samples(*args,**kwargs)
        mimic_samples = self._tf_to_mimic_samples(samples)
        return mimic_samples
    
    def set_dimension(self, dimension):
        """
        Reset the dimension of the problem.
        Calls DiscreteDistribution.set_dimension
        Args:
            dimension (int): new dimension
        Note:
            monitoring times are evenly spaced as linspace(1/dimension,1,dimension)
        """
        self.distribution.set_dimension(dimension)
        self.d = dimension
        self.time_vector = linspace(1./self.d,1,self.d)
        self.ms_vec = self.mean_shift_is * self.time_vector
        sigma = array([[min(self.time_vector[i],self.time_vector[j])
                        for i in range(self.d)]
                        for j in range(self.d)])
        self.a = cholesky(sigma).T
