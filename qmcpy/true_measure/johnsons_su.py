from ._true_measure import TrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
from numpy import *
from scipy.stats import norm


class JohnsonsSU(TrueMeasure):
    """
    >>> jsu = JohnsonsSU(DigitalNetB2(2,seed=7),gamma=1,xi=2,delta=3,lam=4)
    >>> jsu.gen_samples(4)
    array([[ 0.86224892, -0.76967276],
           [ 0.07317047,  1.17727769],
           [ 1.89093286,  2.9341619 ],
           [-1.30283298,  0.62269632]])
    >>> jsu
    JohnsonsSU (TrueMeasure Object)
        gamma           1
        xi              2^(1)
        delta           3
        lam             2^(2)
    
    See https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution
    """

    def __init__(self, sampler, gamma=1, xi=1, delta=2, lam=2):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            gamma (ndarray): gamma
            xi (ndarray): xi
            delta (ndarray): delta > 0
            lam (ndarray): lambda > 0
        """
        self.parameters = ['gamma', 'xi', 'delta', 'lam']
        self.domain = array([[0,1]])
        self.range = array([[-inf,inf]])
        self._parse_sampler(sampler)
        self.gamma = gamma
        self.xi = xi
        self.delta = delta
        self.lam = lam
        if isscalar(self.gamma):
            gamma = tile(self.gamma,self.d)
        if isscalar(self.xi):
            xi = tile(self.xi,self.d)
        if isscalar(self.delta):
            delta = tile(self.delta,self.d)
        if isscalar(self.lam):
            lam = tile(self.lam,self.d)
        self._gamma = array(gamma)
        self._xi = array(xi)
        self._delta = array(delta)
        self._lam = array(lam)
        if len(self._gamma)!=self.d or len(self._xi)!=self.d or len(self._delta)!=self.d or len(self._lam)!=self.d:
            raise DimensionError("all Johnson's S_U parameters be scalar or have length equal to dimension.")
        if (self._delta<=0).any() or (self._lam<=0).any():
            raise ParameterError("delta and lam (lambda) must be greater than 0")
        super(JohnsonsSU,self).__init__() 

    def _transform(self, x):
        return self._lam*sinh((norm.ppf(x)-self._gamma)/self._delta)+self._xi
    
    def _weight(self, x):
        term1 = (x-self._xi)/self._lam
        term2 = self._delta/(self._lam*sqrt(2*pi)) * 1/sqrt(1+term1**2)
        term3 = exp(-1/2*(self._gamma+self._delta*arcsinh(term1))**2)
        return prod( term2*term3, 1)
    
    def _spawn(self, sampler, dimension):
        if dimension==self.d: # don't do anything if the dimension doesn't change
            spawn = JohnsonsSU(sampler,gamma=self._gamma,xi=self._xi,delta=self._delta,lam=self._lam)
        else:
            gamma = self._gamma[0]
            xi = self._xi[0]
            delta = self._delta[0]
            lam = self._lam[0]
            if not ( all(self._gamma==gamma) and all(self._xi==xi) and all(self._delta==delta) and all(self._lam==lam) ):
                raise DimensionError('''
                    In order to spawn a JohnsonsSU measure
                    gamma must all be the same and 
                    xi must all be the same and 
                    delta must all be the same and 
                    lam (lambda) must all be the same.''')
            spawn = JohnsonsSU(sampler,gamma=gamma,xi=xi,delta=delta,lam=lam)
        return spawn
