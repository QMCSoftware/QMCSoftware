from ._true_measure import TrueMeasure
from ..discrete_distribution import Sobol
from ..util import ParameterError


class IdentitalToDiscrete(TrueMeasure):
    """
    For when Discrete Distribution samples already mimic the TrueMeasure. 
    AKA: when g, the original integrand, and f, the transformed integrand are such that: 
    g(x) = f(x) for x ~ DiscreteDistribution. 
    
    >>> dd = Sobol(2,seed=7,randomize=False,graycode=False)
    >>> itd = IdentitalToDiscrete(dd)
    >>> itd.gen_samples(4)
    array([[0.  , 0.  ],
           [0.5 , 0.5 ],
           [0.25, 0.75],
           [0.75, 0.25]])
    """

    parameters = []

    def __init__(self, distribution):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
        """
        self.distribution = distribution
        super(IdentitalToDiscrete,self).__init__()

    def _transform_g_to_f(self, g):
        """ See abstract method. """
        f = lambda samples, *args, **kwargs: g(samples, *args, **kwargs)
        return f
    
    def gen_samples(self, *args, **kwargs):
        """ See abstract method. """
        return self.distribution.gen_samples(*args,**kwargs)

    def plot(self, *args, **kwargs):
        """ Throw error if trying to plot. """
        raise ParameterError('No plotting method for IdenticalToDiscrete. ' +\
            'Try plotting %s directly.'%type(self.distribution).__name__)
