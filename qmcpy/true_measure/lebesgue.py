from ._true_measure import TrueMeasure
from ..discrete_distribution import Sobol
from ..util import TransformError, ParameterError
from numpy import array, isfinite, inf, isscalar, tile
from scipy.stats import norm


class Lebesgue(TrueMeasure):
    """
    >>> dd = Sobol(2,seed=7)
    >>> Lebesgue(dd,lower_bound=[-1,0],upper_bound=[1,3])
    Lebesgue (TrueMeasure Object)
        lower_bound     [-1  0]
        upper_bound     [1 3]
    >>> Lebesgue(dd,lower_bound=-inf,upper_bound=inf)
    Lebesgue (TrueMeasure Object)
        lower_bound     [-inf -inf]
        upper_bound     [inf inf]
    """

    parameters = ['lower_bound', 'upper_bound']
    
    def __init__(self, distribution, lower_bound=0., upper_bound=1.):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
            lower_bound (float or inf): lower bound of integration
            upper_bound (float or inf): upper bound of integration
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
        if isfinite(self.lower_bound).all() and isfinite(self.upper_bound).all() \
           and (self.lower_bound<self.upper_bound).all():
            self.tf_to_mimic = 'StdUniform'
        elif (self.lower_bound == -inf).all() and (self.upper_bound == inf).all():
            self.tf_to_mimic = 'StdGaussian'
        else:
            raise ParameterError('self.lower_bound and self.upper_bound must both be finite ' + \
                                 'or must be -inf,inf respectively')
        super(Lebesgue,self).__init__()

    def _tf_to_mimic_samples(self, samples):
        """
        Transform samples to appear standard uniform or standard gaussian
        
        Args:
            samples (ndarray): samples from a discrete distribution
        
        Return:
            ndarray: transformed samples from the DiscreteDistribution.
        """
        if (self.tf_to_mimic == 'StdUniform' and self.distribution.mimics == 'StdUniform') or \
           (self.tf_to_mimic == 'StdGaussian' and self.distribution.mimics == 'StdGaussian'):
            mimic_samples = samples # identity
        elif self.tf_to_mimic == 'StdUniform' and self.distribution.mimics == 'StdGaussian':
            mimic_samples = norm.cdf(samples) # cdf normal 
        elif self.tf_to_mimic == 'StdGaussian' and self.distribution.mimics == 'StdUniform':
            mimic_samples = norm.ppf(samples) # inverse cdf normal
        else:
            raise TransformError(\
                'Cannot transform samples mimicing %s to mimic %s'%(self.distribution.mimics,self.tf_to_mimic))
        return mimic_samples
    
    def _transform_g_to_f(self, g):
        """ See abstract method. """
        if self.tf_to_mimic == 'StdUniform':
            def f(samples, *args, **kwargs):
                mimic_smaples = self._tf_to_mimic_samples(samples)
                dist = self.upper_bound - self.lower_bound
                f_vals = dist.prod() * g(dist*mimic_smaples + self.lower_bound, *args, **kwargs)
                return f_vals
        elif self.tf_to_mimic == 'StdGaussian':
            def f(samples, *args, **kwargs):
                mimic_smaples = self._tf_to_mimic_samples(samples)
                g_vals = g(mimic_smaples, *args, **kwargs)
                f_vals = g_vals / norm.pdf(mimic_smaples).prod(-1).reshape(g_vals.shape)
                return f_vals
        return f
