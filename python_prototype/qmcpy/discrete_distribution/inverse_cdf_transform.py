""" Definition of InverseCDFTransform, a concrete implementation of DiscreteDistribution """

from ._discrete_distribution import DiscreteDistribution
from ..util import TransformError


class InverseCDFTransform(DiscreteDistribution):
    """ Custom InverseCDFTransform DiscreteDistribution """

    parameters = []

    def __init__(self, distribution_mimicing_uniform, inverse_cdf_fun=lambda u: u):
        """
        Args:
            distribution_mimicing_uniform (DiscreteDistribution): DiscreteDistribution instance
            inverse_cdf_fun (function): function accepting samples mimicing uniform
                                    and applying inverse CDF transform
                Args:
                    u (ndarray): nxd numpy array of samples mimicing uniform
        Example of exponential distribution:
            y ~ exp(l)
            ==> pdf y f(x) = l*exp(-l*x) ==> cdf y F(x)= 1-exp(-l*x)
            ==> F^(-1)(u) = log(1-u)/(-l) ~ exp(l) for u ~ Uniform(0,1)
            ==> inverse_cdf_fun = lambda u,l=5: log(1-u)/(-l)
        """
        self.distribution_u = distribution_mimicing_uniform
        self.inverse_cdf_fun = inverse_cdf_fun
        if self.distribution_u.mimics != 'StdUniform':
            raise TransformError(\
                'Can only apply inverse CDF transform to DiscreteDistributions mimicing StdUniform')
        self.dimension = self.distribution_u.dimension
        self.mimics = 'None'
        super().__init__()
    
    def gen_samples(self, n):
        """
        Generate n x self.dimension samples 

        Args:
            n (int): Number of observations to generate

        Returns:
            n x self.dimension (ndarray)
        """
        original_samples = self.distribution_u.gen_samples(n)
        mimic_samples = self.inverse_cdf_fun(original_samples)
        return mimic_samples
