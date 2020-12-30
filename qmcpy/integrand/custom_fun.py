from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian

class CustomFun(Integrand):
    """
    Custom user-supplied function handle. 
    
    >>> dd = Sobol(2,seed=7)
    >>> m = Gaussian(dd,mean=[1,2])
    >>> cf = CustomFun(m,lambda x: x[:,0]**2*x[:,1])
    >>> x = dd.gen_samples(2**10)
    >>> y = cf.f(x)
    >>> y.mean()
    4.00...
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
