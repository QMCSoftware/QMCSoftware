from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Uniform
from numpy import *


class Linear0(Integrand):
    """
    >>> dd = Sobol(100,seed=7)
    >>> m = Gaussian(dd,mean=(-1)**arange(100),covariance=1./3)
    >>> l = Linear0(m)
    >>> x = dd.gen_samples(2**10)
    >>> y = l.f(x)
    >>> y.mean()
    -0.0003...
    """

    def __init__(self, discrete_distrib):
        """
        Args:
            discrete_distrib (DiscreteDistribution): a discrete distribution instance.
        """
        self.discrete_distrib = discrete_distrib
        self.true_measure = Uniform(self.discrete_distrib.d, lower_bound=-.5, upper_bound=.5)
        super(Linear0,self).__init__()

    def g(self, x):
        y = x.sum(1)
        return y

