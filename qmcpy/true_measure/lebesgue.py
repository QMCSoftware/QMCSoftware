from ._true_measure import TrueMeasure
from ..util import TransformError, ParameterError
from numpy import array, isfinite, inf, isscalar, tile
from scipy.stats import norm


class Lebesgue(TrueMeasure):

    parameters = ['lower_bound', 'upper_bound']
    
    def __init__(self, distribution, lower_bound=0., upper_bound=1):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
            lower_bound (float or inf): lower bound of integration
            upper_bound (float or inf): upper bound of integration
        """
        self.distribution = distribution
        self.d = self.distribution.dimension
        if not self.distribution.mimics == "StdUniform":
            raise ParameterError("Lebesgue measure requires a distribution mimicing 'StdUniform'")        
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
            self.tf_to_mimic = 'Uniform'
        elif (self.lower_bound == -inf).all() and (self.upper_bound == inf).all():
            self.tf_to_mimic = 'Gaussian'
        else:
            raise ParameterError('self.lower_bound and self.upper_bound must both be finite ' + \
                                 'or must be -inf,inf respectively')
        super().__init__()

    def transform_g_to_f(self, g):
        """ See abstract method. """
        if self.tf_to_mimic == 'Uniform':
            def f(samples, *args, **kwargs):
                dist = self.upper_bound - self.lower_bound
                f_vals = dist.prod() * g(dist*samples + self.lower_bound, *args, **kwargs)
                return f_vals
        elif self.tf_to_mimic == 'Gaussian':
            def f(samples, *args, **kwargs):
                inv_cdf_vals = norm.ppf(samples)
                g_vals = g(inv_cdf_vals, *args, **kwargs)
                f_vals = g_vals / norm.pdf(inv_cdf_vals).prod(-1).reshape(g_vals.shape)
                return f_vals
        return f
