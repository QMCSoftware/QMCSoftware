from ._true_measure import TrueMeasure
from ..util import TransformError, DimensionError
from numpy import array, prod, isscalar, tile
from scipy.stats import norm


class Uniform(TrueMeasure):

    parameters = ['lower_bound', 'upper_bound']
    
    def __init__(self, distribution, lower_bound=0., upper_bound=1.):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
            lower_bound (float): a for Uniform(a,b)
            upper_bound (float): b for Uniform(a,b)
        """
        self.distribution = distribution
        self.d = self.distribution.dimension
        if isscalar(lower_bound):
            lower_bound = tile(lower_bound,self.d)
        if isscalar(upper_bound):
            upper_bound = tile(upper_bound,self.d)
        self.lower_bound = array(lower_bound)
        self.upper_bound = array(upper_bound)
        if len(self.lower_bound)!=self.d or len(self.upper_bound)!=self.d:
            raise DimensionError('upper bound and lower bound must be of length dimension')
        super().__init__()
    
    def pdf(self,x):
        """ See abstract class. """
        return 1/( prod(self.upper_bound-self.lower_bound) )

    def _tf_to_mimic_samples(self, samples):
        """
        Transform samples to appear Uniform
        
        Args:
            samples (ndarray): samples from a discrete distribution
        
        Return:
            ndarray: samples from the DiscreteDistribution transformed to mimic Uniform.
        """
        if self.distribution.mimics == 'StdGaussian':
            # CDF then stretch
            mimic_samples = norm.cdf(samples) * (self.upper_bound - self.lower_bound) + self.lower_bound
        elif self.distribution.mimics == "StdUniform":
            # stretch samples
            mimic_samples = samples * (self.upper_bound - self.lower_bound) + self.lower_bound
        else:
            raise TransformError(\
                'Cannot transform samples mimicing %s to Uniform'%self.distribution.mimics)
        return mimic_samples

    def transform_g_to_f(self, g):
        """ See abstract method. """
        def f(samples, *args, **kwargs):
            z = self._tf_to_mimic_samples(samples)
            y = g(z, *args, **kwargs)
            return y
        return f
        return f
    
    def gen_mimic_samples(self, *args, **kwargs):
        """ See abstract method. """
        samples = self.distribution.gen_samples(*args,**kwargs)
        mimic_samples = self._tf_to_mimic_samples(samples)
        return mimic_samples
    
    def set_dimension(self, dimension):
        """ See abstract method. """
        l = self.lower_bound[0]
        u = self.upper_bound[0]
        if not (all(self.lower_bound==l) and all(self.upper_bound==u)):
            raise DimensionError('In order to change dimension of uniform measure the '+\
                'lower bounds must all be the same and the upper bounds must all be the same')
        self.distribution.set_dimension(dimension)
        self.d = dimension
        self.lower_bound = tile(l,self.d)
        self.upper_bound = tile(u,self.d)
