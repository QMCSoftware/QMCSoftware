from ._integrand import Integrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Uniform


class Linear0(Integrand):
    """
    >>> l = Linear0(DigitalNetB2(100,seed=7))
    >>> x = l.discrete_distrib.gen_samples(2**10)
    >>> y = l.f(x)
    >>> y.mean().item()
    3.0517083357897278e-05
    >>> ytf = l.f(x,periodization_transform='C1SIN')
    >>> ytf.mean().item()
    1.4009858752211625e-14
    """

    def __init__(self, sampler):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
        """
        self.sampler = sampler
        self.true_measure = Uniform(self.sampler, lower_bound=-.5, upper_bound=.5)
        super(Linear0,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)
    
    def g(self, t):
        y = t.sum(1)
        return y

    def _spawn(self, level, sampler):
        return Linear0(sampler=sampler)
