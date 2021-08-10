from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian

class CustomFun(Integrand):
    """
    Custom user-supplied function handle. 
    
    >>> cf = CustomFun(
    ...     true_measure = Gaussian(Sobol(2,seed=7),mean=[1,2]),
    ...     g = lambda x: x[:,0]**2*x[:,1])
    >>> x = cf.discrete_distrib.gen_samples(2**10)
    >>> y = cf.f(x)
    >>> y.mean()
    4.001...
    """

    def __init__(self, true_measure, g):
        """
        Args:
            true_measure (TrueMeasure): a TrueMeasure instance. 
            g (function): a function handle. 
        """
        self.parameters = []
        self.true_measure = true_measure
        self._g = g
        self.dprime = 1
        super(CustomFun,self).__init__()
    
    def g(self, t, *args, **kwargs):
        return self._g(t,*args,**kwargs)

