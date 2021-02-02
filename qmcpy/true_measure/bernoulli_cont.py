from qmcpy.true_measure._true_measure import TrueMeasure
from qmcpy.util import DimensionError
from ..discrete_distribution import Sobol
from numpy import *


class BernoulliCont(TrueMeasure):
    """
    Continuous Bernoulli Measure.
    """

    def __init__(self, sampler, lam=1/2, b=2):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            lam (ndarray): lambda, a shape parameter, independent for each dimension 
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
            tf[:,j] = x[:,j] if self.l[j]==1/2 else (log((2*self.l[j]-1)*x[:,j]-self.l[j]+1)-log(1-self.l[j])) / log(self.l[j]/(1-self.l[j]))
        return tf

    def _jacobian(self, x):
        j = zeros(x.shape,dtype=float)
        for j in range(self.d):
            j[:,j] = 1 if self.l[j]==1/2 else 1/log(self.l[j]/(1-self.l[j])) * (2*self.l[j]-1)/((2*self.l[j]-1)*x[:,j]-self.l[j]+1)
        return prod(j,1)
    
    def _weight(self, x):
        w = zeros(x.shape,dtype=float)
        for j in range(self.d):
            C = 2 if self.l[j]==1/2 else 2*arctanh(1-2*self.l[j])/(1-2*self.l[j])
            w[:,j] = C*self.l[j]**x[:,j]*(1-self.l[j])**(1-x[:,j])
        return prod(w,1)
    
    def _set_dimension(self, dimension):
        l0 = self.l[0]
        if (self.l!=l0).any():
            raise DimensionError('''
                In order to change dimension of a Continuous Bernoulli measure
                lam must all be the same.''')
        self.d = dimension
        self.l = tile(l0,self.d)