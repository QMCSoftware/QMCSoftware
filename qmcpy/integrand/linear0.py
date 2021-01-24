from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Uniform


class Linear0(Integrand):
    """
    >>> l = Linear0(Sobol(100,seed=7))
    >>> x = l.discrete_distrib.gen_samples(2**10)
    >>> y = l.f(x)
    >>> y.mean()
    -2.654...e-08
    >>> ytf = l.f_periodized(x,'C1SIN')
    >>> ytf.mean()
    8.288...e-10
    """

    def __init__(self, sampler):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
        """
        self.true_measure = Uniform(sampler, lower_bound=-.5, upper_bound=.5)
        super(Linear0,self).__init__()

    def g(self, x):
        y = x.sum(1)
        return y

