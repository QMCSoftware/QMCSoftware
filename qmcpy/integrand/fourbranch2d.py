import numpy as np
from ._integrand import Integrand
from ..true_measure import Uniform
from ..discrete_distribution import DigitalNetB2

class FourBranch2d(Integrand):
    """
    >>> fb2d = FourBranch2d(DigitalNetB2(2,seed=7))
    >>> x = fb2d.discrete_distrib.gen_samples(2**10)
    >>> y = fb2d.f(x)
    >>> y.mean().item()
    -2.500835871323173
    >>> fb2d.true_measure
    Uniform (TrueMeasure Object)
        lower_bound     -8
        upper_bound     2^(3)
    """
    def __init__(self, sampler):
        self.sampler = sampler
        assert self.sampler.d==2
        self.true_measure = Uniform(self.sampler,lower_bound=-8,upper_bound=8)
        super(FourBranch2d,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)
    def g(self, t):
        t0,t1 = t[:,0],t[:,1]
        return np.vstack([
            3+.1*(t0-t1)**2-(t0+t1)/np.sqrt(2),
            3+.1*(t0-t1)**2+(t0+t1)/np.sqrt(2),
            t0-t1+7/np.sqrt(2),
            t1-t0+7/np.sqrt(2)]).min(0)
    def _spawn(self, level, sampler):
        return FourBranch2d(sampler=sampler) 
