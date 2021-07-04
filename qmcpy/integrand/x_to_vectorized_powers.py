from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Uniform
from numpy import *


class XtoVectorizedPowers(Integrand):
    """
    >>> l1 = XtoVectorizedPowers(Sobol(1,seed=7), powers=[3])
    >>> x1 = l1.discrete_distrib.gen_samples(2**10)
    >>> y1 = l1.f(x1,compute_flags=[1])
    >>> y1.shape
    (1024, 1)
    >>> y1.mean(0)
    array([0.25])
    >>> l2 = XtoVectorizedPowers(Sobol(2,seed=7), powers=[4,5])
    >>> x2 = l2.discrete_distrib.gen_samples(2**10)
    >>> y2 = l2.f(x2,compute_flags=[1,1])
    >>> y2.shape
    (1024, 2)
    >>> y2.mean(0)
    array([0.4  , 0.333])
    """

    def __init__(self, sampler, powers=array([1,2])):
        """
        For (n by d)  sample x and k-vector powers = (p_0,...,p_{k-1}), 
        return [
            [x_{0,0}^p_0+x_{0,1}^p_0+...+x_{0,d-1}^p_0], ..., x_{0,0}^p_{k-1}+x_{0,1}^p_{k-1}+...+x_{0,d-1}^p_{k-1}],
            ...
            [[x_{n-1,0}^p_0+x_{n-1,1}^p_0+...+x_{n-1,d-1}^p_0], ..., x_{n-1,0}^p_{k-1}+x_{n-1,1}^p_{k-1}+...+x_{n-1,d-1}^p_{k-1}]]
        ]
        Essentially, if we return an (n by k) output Y, then Y[i,j] is sum(x[i,:]^powers[j])


        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
        """
        self.powers = array(powers)
        self.k = len(powers)
        self.true_measure = Uniform(sampler, lower_bound=0., upper_bound=1.)
        self.dprime = self.k
        super(XtoVectorizedPowers,self).__init__() # output dimensions per sample

    def g(self, t, compute_flags):
        n,d = t.shape
        Y = zeros((n,self.k),dtype=float)
        for j in range(self.k):
            if compute_flags[j] == 1: 
                Y[:,j] = (t**self.powers[j]).sum(1)
        return Y

