from ._integrand import Integrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Uniform
from numpy import *


class BoxIntegral(Integrand):
    """
    $B_s(x) = \\left(\\sum_{j=1}^d x_j^2 \\right)^{s/2}$

    >>> l1 = BoxIntegral(DigitalNetB2(2,seed=7), s=[7])
    >>> x1 = l1.discrete_distrib.gen_samples(2**10)
    >>> y1 = l1.f(x1)
    >>> y1.shape
    (1024, 1)
    >>> y1.mean(0)
    array([0.75156724])
    >>> l2 = BoxIntegral(DigitalNetB2(5,seed=7), s=[-7,7])
    >>> x2 = l2.discrete_distrib.gen_samples(2**10)
    >>> y2 = l2.f(x2,compute_flags=[1,1])
    >>> y2.shape
    (1024, 2)
    >>> y2.mean(0)
    array([ 6.67548708, 10.52267786])

    References:

    [1] D.H. Bailey, J.M. Borwein, R.E. Crandall,Box integrals,
    Journal of Computational and Applied Mathematics, Volume 206, Issue 1, 2007, Pages 196-208, ISSN 0377-0427, 
    https://doi.org/10.1016/j.cam.2006.06.010. (https://www.sciencedirect.com/science/article/pii/S0377042706004250) 
    
    [2] https://www.davidhbailey.com/dhbpapers/boxintegrals.pdf
    """

    def __init__(self, sampler, s=array([1,2])):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
            s (list or ndarray): vectorized s parameter, len(s) is the number of vectorized integrals to evalute.
        """
        self.parameters = ['s']
        self.s = array([s]) if isscalar(s) else array(s)
        self.sampler = sampler
        self.true_measure = Uniform(self.sampler, lower_bound=0., upper_bound=1.)
        super(BoxIntegral,self).__init__(dprime=len(self.s),parallel=False) # output dimensions per sample

    def g(self, t, **kwargs):
        compute_flags = kwargs['compute_flags'] if 'compute_flags' in kwargs else ones(self.dprime,dtype=int)
        n,d = t.shape
        y = zeros((n,)+self.dprime,dtype=float)
        for j in range(len(self.s)):
            if compute_flags[j] == 1: 
                y[:,j] = (t**2).sum(1)**(self.s[j]/2)
        return y
    
    def _spawn(self, level, sampler):
        return BoxIntegral(sampler=sampler,s=self.s)
