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
    4.0092435574693095
    """

    parameters = []

    def __init__(self, true_measure, g):
        """
        Args:
            true_measure (TrueMeasure): a TrueMeasure instance. 
            g (function): a function handle. 
        """
        self.true_measure = true_measure
        self.g = lambda x,*args,**kwargs: g(x,*args,**kwargs) 
        super(CustomFun,self).__init__()
