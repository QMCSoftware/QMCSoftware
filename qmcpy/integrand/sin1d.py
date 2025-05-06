import numpy as np
from ._integrand import Integrand
from ..true_measure import Uniform
from ..discrete_distribution import DigitalNetB2

class Sin1d(Integrand):
    """
    >>> sin1d = Sin1d(DigitalNetB2(1,seed=7))
    >>> x = sin1d.discrete_distrib.gen_samples(2**10)
    >>> y = sin1d.f(x)
    >>> print("%.4e"%y.mean())
    -7.3732e-08
    >>> sin1d.true_measure
    Uniform (AbstractTrueMeasure Object)
        lower_bound     0
        upper_bound     6.283
    """
    def __init__(self, sampler, k=1):
        self.sampler = sampler
        self.k = k
        assert self.sampler.d==1
        self.true_measure = Uniform(self.sampler,lower_bound=0,upper_bound=2*self.k*np.pi)
        super(Sin1d,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)
    def g(self, t):
        return np.sin(t.squeeze())
    def _spawn(self, level, sampler):
        return Sin1d(sampler=sampler,k=self.k)
