from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Uniform
from numpy import *


class BoxIntegral(Integrand):
    """
    >>> l1 = BoxIntegral(Sobol(2,seed=7), s=[7])
    >>> x1 = l1.discrete_distrib.gen_samples(2**10)
    >>> y1 = l1.f(x1)
    >>> y1.shape
    (1024, 1)
    >>> y1.mean(0)
    array([0.75165595])
    >>> l2 = BoxIntegral(Sobol(5,seed=7), s=[-7,7])
    >>> x2 = l2.discrete_distrib.gen_samples(2**10)
    >>> y2 = l2.f(x2,compute_flags=[1,1])
    >>> y2.shape
    (1024, 2)
    >>> y2.mean(0)
    array([ 6.75294324, 10.52129649])
    """

    def __init__(self, sampler, s=array([1,2])):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
        
        https://www.davidhbailey.com/dhbpapers/boxintegrals.pdf
        """
        self.parameters = ['s']
        self.s = array(s)
        self.dprime = len(self.s)
        self.true_measure = Uniform(sampler, lower_bound=0., upper_bound=1.)
        super(BoxIntegral,self).__init__() # output dimensions per sample

    def g(self, t, **kwargs):
        compute_flags = kwargs['compute_flags'] if 'compute_flags' in kwargs else ones(self.dprime,dtype=int)
        n,d = t.shape
        Y = zeros((n,self.dprime),dtype=float)
        for j in range(self.dprime):
            if compute_flags[j] == 1: 
                Y[:,j] = (t**2).sum(1)**(self.s[j]/2)
        return Y
