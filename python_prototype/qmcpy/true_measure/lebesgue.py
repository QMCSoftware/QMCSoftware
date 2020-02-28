""" Definition of Lebesgue, a concrete implementation of TrueMeasure """

from ._true_measure import TrueMeasure
from ..util import TransformError, ParameterError
from numpy import array, isfinite, inf
from scipy.stats import norm


class Lebesgue(TrueMeasure):
    """ Lebesgue Uniform TrueMeasure """

    parameters = ['lower_bound', 'upper_bound']
    
    def __init__(self, distribution, lower_bound=0., upper_bound=1):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
            lower_bound (float or inf): lower bound of integration
            upper_bound (float or inf): upper bound of integration
        """
        self.distribution = distribution
        if not self.distribution.mimics == "StdUniform":
            raise ParameterError("Lebesgue measure requires a distribution mimicing 'StdUniform'")
        self.lower_bound = array(lower_bound)
        self.upper_bound = array(upper_bound)
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
        """
        Transform g, the origianl integrand, to f,
        the integrand accepting standard distribution sampels. 
        
        Args:
            g (method): original integrand
        
        Returns:
            f (method): transformed integrand
        """
        if self.tf_to_mimic == 'Uniform':
            def f(samples):
                dist = self.upper_bound - self.lower_bound
                vals = dist.prod() * g(dist*samples + self.lower_bound)
                return vals
        elif self.tf_to_mimic == 'Gaussian':
            def f(samples):
                inv_cdf_vals = norm.ppf(samples)
                vals = g(inv_cdf_vals) / norm.pdf(inv_cdf_vals)

        return f
    
    def gen_mimic_samples(self, *args, **kwargs):
        raise Exception("Cannot generate samples mimicking a Lebesgue measure")
