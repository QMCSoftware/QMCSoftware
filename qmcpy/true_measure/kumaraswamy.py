from ._true_measure import TrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
from numpy import *


class Kumaraswamy(TrueMeasure):
    """
    >>> k = Kumaraswamy(DigitalNetB2(2,seed=7),a=[1,2],b=[3,4])
    >>> k.gen_samples(4)
    array([[0.24096272, 0.21587652],
           [0.13227662, 0.4808615 ],
           [0.43615893, 0.73428949],
           [0.03602294, 0.39602319]])
    >>> k
    Kumaraswamy (TrueMeasure Object)
        a               [1 2]
        b               [3 4]
        
    See https://en.wikipedia.org/wiki/Kumaraswamy_distribution
    """

    def __init__(self, sampler, a=2, b=2):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            a (ndarray): alpha > 0
            b (ndarray): beta > 0
        """
        self.parameters = ['a', 'b']
        self.domain = array([[0,1]])
        self.range = array([[0,1]])
        self._parse_sampler(sampler)
        self.a = a
        self.b = b
        if isscalar(self.a):
            a = tile(self.a,self.d)
        if isscalar(self.b):
            b = tile(self.b,self.d)
        self.alpha = array(a)
        self.beta = array(b)
        if len(self.alpha)!=self.d or len(self.beta)!=self.d:
            raise DimensionError('a and b must be scalar or have length equal to dimension.')
        if not (all(self.alpha>0) and all(self.beta>0)):
            raise ParameterError("Kumaraswamy requires a,b>0.")
        super(Kumaraswamy,self).__init__() 

    def _transform(self, x):
        return (1-(1-x)**(1/self.beta))**(1/self.alpha)
    
    def _weight(self, x):
        return prod( self.alpha*self.beta*x**(self.alpha-1)*(1-x**self.alpha)**(self.beta-1), 1)
    
    def _spawn(self, sampler, dimension):
        if dimension==self.d: # don't do anything if the dimension doesn't change
            spawn = Kumaraswamy(sampler,a=self.alpha,b=self.beta)
        else:
            a = self.alpha[0]
            b = self.beta[0]
            if not (all(self.alpha==a) and all(self.beta==b)):
                raise DimensionError('''
                    In order to spawn a Kumaraswamy measure
                    a must all be the same and 
                    b must all be the same''')
            spawn = Kumaraswamy(sampler,a=a,b=b)
        return spawn