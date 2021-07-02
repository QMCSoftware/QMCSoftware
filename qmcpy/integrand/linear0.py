from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Uniform


class Linear0(Integrand):
    """
    >>> l = Linear0(Sobol(100,seed=7))
    >>> x = l.discrete_distrib.gen_samples(2**10)
    >>> y = l.f(x)
    >>> y.mean()
    -7.378...e-06
    >>> ytf = l.f_periodized(x,'C1SIN')
    >>> ytf.mean()
    9.037...e-12
    """

    def __init__(self, sampler):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
        """
        self.true_measure = Uniform(sampler, lower_bound=-.5, upper_bound=.5)
        super(Linear0,self).__init__(output_dims=1)
    
    def g(self, t):
        y = t.sum(1)
        return y

