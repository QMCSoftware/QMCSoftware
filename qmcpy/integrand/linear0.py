from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Uniform


class Linear0(Integrand):
    """
    >>> l = Linear0(Sobol(100,seed=7))
    >>> x = l.discrete_distrib.gen_samples(2**10)
    >>> y = l.f(x)
    >>> y.mean()
    -2.6542693376541138e-08
    >>> ytf = l.f_periodized(x,'C1SIN')
    >>> ytf.mean()
    8.288376592929427e-10
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

