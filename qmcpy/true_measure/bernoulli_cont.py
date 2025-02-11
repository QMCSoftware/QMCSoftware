from ._true_measure import TrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
from numpy import *


class BernoulliCont(TrueMeasure):
    """    
    >>> bc = BernoulliCont(DigitalNetB2(2,seed=7),lam=.2)
    >>> bc.gen_samples(4)
    array([[0.03979019, 0.04339218],
           [0.68043258, 0.58998885],
           [0.1937716 , 0.86669211],
           [0.40386874, 0.16007927]])
    >>> bc
    BernoulliCont (TrueMeasure Object)
        lam             0.200
    
    See https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution
    """

    def __init__(self, sampler, lam=1/2):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            lam (ndarray): 0 < lambda < 1, a shape parameter, independent for each dimension 
        """
        self.parameters = ['lam']
        self.domain = array([[0,1]])
        self.range = array([[0,1]])
        self._parse_sampler(sampler)
        self.lam = lam
        if isscalar(self.lam):
            lam = tile(self.lam,self.d)
        self.l = array(lam)
        if len(self.l)!=self.d or (self.l<=0).any() or (self.l>=1).any():
            raise DimensionError('lam must be scalar or have length equal to dimension and must be in (0,1).')
        super(BernoulliCont,self).__init__() 

    def _transform(self, x):
        tf = zeros(x.shape,dtype=float)
        for j in range(self.d):
            tf[:,j] = x[:,j] if self.l[j]==1/2 else log(((2*self.l[j]-1)*x[:,j]-self.l[j]+1)/(1-self.l[j])) / log(self.l[j]/(1-self.l[j]))
        return tf
    
    def _weight(self, x):
        w = zeros(x.shape,dtype=float)
        for j in range(self.d):
            C = 2 if self.l[j]==1/2 else 2*arctanh(1-2*self.l[j])/(1-2*self.l[j])
            w[:,j] = C*self.l[j]**x[:,j]*(1-self.l[j])**(1-x[:,j])
        return prod(w,1)
    
    def _spawn(self, sampler, dimension):
        if dimension==self.d: # don't do anything if the dimension doesn't change
            spawn = BernoulliCont(sampler,lam=self.lam)
        else:
            l0 = self.l[0]
            if (self.l!=l0).any():
                raise DimensionError('''
                        In order to spawn a BernoulliCont measure
                        lam must all be the same.''')
            spawn = BernoulliCont(sampler,lam=l0)
        return spawn
