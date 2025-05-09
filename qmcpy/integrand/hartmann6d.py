import numpy as np
from .abstract_integrand import AbstractIntegrand
from ..true_measure import Uniform
from ..discrete_distribution import DigitalNetB2

class Hartmann6d(AbstractIntegrand):
    r"""
    Wrapper around [`BoTorch`'s implementation of the Augmented Hartmann function](https://botorch.readthedocs.io/en/stable/test_functions.html#botorch.test_functions.multi_fidelity.AugmentedHartmann) in dimension $d=6$. 

    Examples:
        >>> integrand = Hartmann6d(DigitalNetB2(6,seed=7))
        >>> y = integrand(2**10)
        >>> print("%.4f"%y.mean())
        -0.2550
        >>> integrand.true_measure
        Uniform (AbstractTrueMeasure)
            lower_bound     0
            upper_bound     1
            
        With independent replications

        >>> integrand = Hartmann6d(DigitalNetB2(6,seed=7,replications=2**4))
        >>> y = integrand(2**6)
        >>> y.shape
        (16, 64)
        >>> muhats = y.mean(-1) 
        >>> muhats.shape 
        (16,)
        >>> print("%.4f"%muhats.mean())
        -0.2556
    """
    
    def __init__(self, sampler):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): Either  
                
                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
        """
        self.sampler = sampler
        assert self.sampler.d==6
        self.true_measure = Uniform(self.sampler,lower_bound=0,upper_bound=1)
        super(Hartmann6d,self).__init__(dimension_indv=(),dimension_comb=(),parallel=False)
        from botorch.test_functions.multi_fidelity import AugmentedHartmann
        self.ah = AugmentedHartmann(negate=False)
        
    def g(self, t):
        import torch
        t = np.concatenate([t,np.ones(tuple(t.shape[:-1])+(1,))],axis=-1)
        return self.ah.evaluate_true(torch.tensor(t)).numpy()
