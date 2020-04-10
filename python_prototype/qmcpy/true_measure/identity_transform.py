"""
Definition of IdentityTransform, a concrete implementation of TrueMeasure
for g the original integral, f the transformed integral: g(x) = f(x) for x~DiscreteDistribution
"""

from ._true_measure import TrueMeasure


class IdentityTransform(TrueMeasure):
    """ Discrete Distribution samples already mimic the TrueMeasure """

    parameters = []

    def __init__(self, distribution):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
        """
        self.distribution = distribution
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
        f = lambda samples: g(samples)
        return f
    
    def gen_mimic_samples(self, *args, **kwargs):
        """
        Generate samples from the DiscreteDistribution object
        and transform them to mimic TrueMeasure samples
        
        Args:
            *args (tuple): Ordered arguments to self.distribution.gen_samples
            **kwrags (dict): Keyword arguments to self.distribution.gen_samples
        
        Returns:
            mimic_samples (ndarray): samples from the DiscreteDistribution object transformed to appear 
                                  to appear like the TrueMeasure object
        """
        return self.distribution.gen_samples(*args,**kwargs)
