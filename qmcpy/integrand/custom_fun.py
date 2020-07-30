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
    4.005918911572021
    """

    parameters = []

    def __init__(self, measure, custom_fun):
        """
        Args:
            measure (TrueMeasure): a TrueMeasure instance
            custom_fun (function): a function evaluating samples (nxd) -> (nx1). See g method.
        """
        self.measure = measure
        self.distribution = self.measure.distribution
        self.custom_fun = custom_fun
        super(CustomFun,self).__init__()

    def g(self, x):
        """ See abstract method. """
        return self.custom_fun(x)
