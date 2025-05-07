from .abstract_integrand import AbstractIntegrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Uniform
from ..util import ParameterError
import numpy as np

class Genz(AbstractIntegrand):
    """
    https://dakota.sandia.gov/sites/default/files/docs/6.17.0-release/user-html/usingdakota/examples/additionalexamples.html?highlight=genz#genz-functions

    >>> for kind_func in ['oscillatory','corner-peak']:
    ...     for kind_coeff in [1,2,3]:
    ...         g = Genz(DigitalNetB2(2,seed=7),kind_func=kind_func,kind_coeff=kind_coeff)
    ...         x = g.discrete_distrib.gen_samples(2**14)
    ...         y = g.f(x)
    ...         mu_hat = y.mean()
    ...         print('%-15s %-3d %.3f'%(kind_func,kind_coeff,mu_hat))
    oscillatory     1   -0.351
    oscillatory     2   -0.380
    oscillatory     3   -0.217
    corner-peak     1   0.713
    corner-peak     2   0.712
    corner-peak     3   0.720
    """

    def __init__(self, sampler, kind_func='oscillatory', kind_coeff=1):
        """
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
            kind_func (str): 'oscillatory' or 'corner-peak'
            kind_coeff (int): 1, 2, or 3 for choice of coefficients 
        """
        self.kind_func = kind_func.lower()
        self.kind_coeff = kind_coeff
        if (self.kind_func not in ['oscillatory','corner-peak']) or (self.kind_coeff not in [1,2,3]):
            raise ParameterError('''
                Genz expects 
                    kind_func in ['oscillatory','corner-peak'] and 
                    kind_coeffs in [1,2,3]''')
        self.sampler = sampler
        self.true_measure = Uniform(self.sampler)
        self.d = self.true_measure.d
        if self.kind_coeff==1: self.c = (np.arange(1,self.d+1)-.5)/self.d
        elif self.kind_coeff==2: self.c = 1/np.arange(1,self.d+1)
        elif self.kind_coeff==3: self.c = np.exp(np.arange(1,self.d+1)*np.log(10**(-8))/self.d)
        if self.kind_func=='oscillatory':
            self.g = self.g_oscillatory
            self.c = 4.5*self.c/self.c.sum()
        elif self.kind_func=='corner-peak':
            self.g = self.g_corner_peak
            self.c = 0.25*self.c/self.c.sum()
        self.c = self.c[None,:]
        self.parameters = ['kind_func','kind_coeff']
        super(Genz,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)
    
    def g_oscillatory(self, t):
        return np.cos(-(self.c*t).sum(1))

    def g_corner_peak(self, t):
        return (1+(self.c*t).sum(1))**(-(self.d+1))
    
    def _spawn(self, level, sampler):
        return Genz(sampler=sampler,kind_func=self.kinda_func,kind_coeff=self.kind_coeff)
