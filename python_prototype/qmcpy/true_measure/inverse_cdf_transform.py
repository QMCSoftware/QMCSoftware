""" Definition of InverseCDFTransform, a concrete implementation of TrueMeasure """

from ._true_measure import TrueMeasure
from ..util import TransformError


class InverseCDFTransform(TrueMeasure):
    """ Custom InverseCDFTransform TrueMeasure """

    parameters = []

    def __init__(self, distribution, inverse_cdf_fun=lambda u: u):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
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
        self.distribution = distribution
        self.inverse_cdf_fun = inverse_cdf_fun
        if self.distribution.mimics != 'StdUniform':
            raise TransformError(\
                'Can only apply inverse CDF transform to DiscreteDistributions mimicing StdUniform')
        super().__init__()
    
    def _tf_to_mimic_samples(self, samples):
        """
        Transform samples to appear Gaussian
        
        Args:
            samples (ndarray): samples from a discrete distribution
        
        Return:
             mimic_samples (ndarray): samples from the DiscreteDistribution transformed to appear 
                                  to appear like the TrueMeasure object
        """
        mimic_samples = self.inverse_cdf_fun(samples)
        return mimic_samples

    def transform_g_to_f(self, g):
        """
        Transform g, the origianl integrand, to f,
        the integrand accepting standard distribution sampels. 
        
        Args:
            g (method): original integrand
        
        Returns:
            f (method): transformed integrand
        """
        f = lambda samples: g(self._tf_to_mimic_samples(samples))
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
