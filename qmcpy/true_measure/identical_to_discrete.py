from ._true_measure import TrueMeasure


class IdentitalToDiscrete(TrueMeasure):
    """
    For when Discrete Distribution samples already mimic the TrueMeasure. 
    AKA: when g, the original integrand, and f, the transformed integrand are such that: 
    g(x) = f(x) for x ~ DiscreteDistribution
    """

    parameters = []

    def __init__(self, distribution):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
        """
        self.distribution = distribution
        super().__init__()

    def transform_g_to_f(self, g):
        """ See abstract method. """
        f = lambda samples, *args, **kwargs: g(samples, *args, **kwargs)
        return f
    
    def gen_mimic_samples(self, *args, **kwargs):
        """ See abstract method. """
        return self.distribution.gen_samples(*args,**kwargs)
