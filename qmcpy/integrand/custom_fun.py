from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian

class CustomFun(Integrand):
    """
    Custom user-supplied function handle. 
    
    >>> d = 2
    >>> s = Sobol(d,seed=7)
    >>> n = Gaussian(d,mean=[1,2])
    >>> g = lambda x: x[:,0]**2*x[:,1]
    >>> cf = CustomFun(s,g,n)
    >>> x = s.gen_samples(2**10)
    >>> y = cf.f(x)
    >>> y.mean()
    4.0092435574693095
    """

    parameters = []

    def __init__(self, discrete_distrib, g, true_measure):
        """
        Args:
            g (function): a function handle. 
            discrete_distrib (DiscreteDistribution): a discrete distribution instance.
            true_measure (TrueMeasure): a TrueMeasure instance. 
        """
        self.discrete_distrib = discrete_distrib
        self.true_measure = true_measure
        self.g = lambda x,*args,**kwargs: g(x,*args,**kwargs) 
        super(CustomFun,self).__init__()
