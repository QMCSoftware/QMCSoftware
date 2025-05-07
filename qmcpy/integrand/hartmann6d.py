import numpy as np
from .abstract_integrand import AbstractIntegrand
from ..true_measure import Uniform
from ..discrete_distribution import DigitalNetB2

class Hartmann6d(AbstractIntegrand):
    """
    >>> h6d = Hartmann6d(DigitalNetB2(6,seed=7))
    >>> x = h6d.discrete_distrib.gen_samples(2**10)
    >>> y = h6d.f(x)
    >>> print("%.4f"%y.mean())
    -0.2591
    >>> h6d.true_measure
    Uniform (AbstractTrueMeasure Object)
        lower_bound     0
        upper_bound     1
    """
    def __init__(self, sampler):
        self.sampler = sampler
        assert self.sampler.d==6
        self.true_measure = Uniform(self.sampler,lower_bound=0,upper_bound=1)
        super(Hartmann6d,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)
        from botorch.test_functions.multi_fidelity import AugmentedHartmann
        self.ah = AugmentedHartmann(negate=False)
        
    def g(self, t):
        import torch
        t = np.hstack([t,np.ones((len(t),1))])
        return self.ah.evaluate_true(torch.tensor(t)).numpy()
