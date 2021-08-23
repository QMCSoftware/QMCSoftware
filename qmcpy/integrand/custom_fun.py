from ._integrand import Integrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Gaussian, Uniform

class CustomFun(Integrand):
    """
    Integrand wrapper for a user's function 
    
    >>> cf = CustomFun(
    ...     true_measure = Gaussian(DigitalNetB2(2,seed=7),mean=[1,2]),
    ...     g = lambda x: x[:,0]**2*x[:,1])
    >>> x = cf.discrete_distrib.gen_samples(2**10)
    >>> y = cf.f(x)
    >>> y.shape
    (1024, 1)
    >>> y.mean()
    3.995...
    >>> cf = CustomFun(
    ...     true_measure = Uniform(DigitalNetB2(3,seed=7),lower_bound=[2,3,4],upper_bound=[4,5,6]),
    ...     g = lambda x: x,
    ...     dprime = 3)
    >>> x = cf.discrete_distrib.gen_samples(2**10)
    >>> y = cf.f(x)
    >>> y.shape
    (1024, 3)
    >>> y.mean(0)
    array([3., 4., 5.])
    """

    def __init__(self, true_measure, g, dprime=1):
        """
        Args:
            true_measure (TrueMeasure): a TrueMeasure instance. 
            g (function): a function handle. 
            dprime (int): output dimension of the function.
        """
        self.parameters = []
        self.true_measure = true_measure
        self._g = g
        self.dprime = dprime
        super(CustomFun,self).__init__()
    
    def g(self, t, *args, **kwargs):
        return self._g(t,*args,**kwargs)

