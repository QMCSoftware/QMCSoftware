from numpy import *
from ._integrand import Integrand
from ..true_measure import Uniform
from ..discrete_distribution import DigitalNetB2

class Hartmann6d(Integrand):
    """
    >>> h6d = Hartmann6d(DigitalNetB2(6,seed=7))
    >>> x = h6d.discrete_distrib.gen_samples(2**10)
    >>> y = h6d.f(x)
    >>> y.mean()
    -0.2613140309713834
    >>> h6d.true_measure
    Uniform (TrueMeasure Object)
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
        t = hstack([t,ones((len(t),1))])
        return self.ah.evaluate_true(torch.tensor(t)).numpy()