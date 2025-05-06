import numpy as np
from ._integrand import Integrand
from ..true_measure import Uniform
from ..discrete_distribution import DigitalNetB2

class Multimodal2d(Integrand):
    """
    >>> mm2d = Multimodal2d(DigitalNetB2(2,seed=7))
    >>> x = mm2d.discrete_distrib.gen_samples(2**10)
    >>> y = mm2d.f(x)
    >>> print("%.4f"%y.mean())
    -0.7375
    >>> mm2d.true_measure
    Uniform (AbstractTrueMeasure Object)
        lower_bound     [-4 -3]
        upper_bound     [7 8]
    """
    def __init__(self, sampler):
        self.sampler = sampler 
        assert self.sampler.d==2
        self.true_measure = Uniform(self.sampler,lower_bound=[-4,-3],upper_bound=[7,8])
        super(Multimodal2d,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)
    def g(self, t):
        t0,t1 = t[:,0],t[:,1]
        return (t0**2+4)*(t1-1)/20-np.sin(5*t0/2)-2
    def _spawn(self, level, sampler):
        return Multimodal2d(sampler=sampler) 
