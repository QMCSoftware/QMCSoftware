from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
import numpy as np
from scipy.stats import norm


class JohnsonsSU(AbstractTrueMeasure):
    """
    Johnson's $S_U$-distribution as described in  
        [https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution](https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution)

    Examples:
        >>> jsu = JohnsonsSU(DigitalNetB2(2,seed=7),gamma=1,xi=2,delta=3,lam=4)
        >>> jsu.gen_samples(4)
        array([[ 2.01636624,  1.44849599],
               [-1.32410385, -1.49239458],
               [ 1.00113995, -0.23987417],
               [ 0.06372867,  2.44847546]])
        >>> jsu
        JohnsonsSU (AbstractTrueMeasure)
            gamma           1
            xi              2^(1)
            delta           3
            lam             2^(2)
        
        With independent replications 

        >>> x = JohnsonsSU(DigitalNetB2(3,seed=7,replications=2),gamma=1,xi=2,delta=3,lam=4)(4)
        >>> x.shape 
        (2, 4, 3)
        >>> x
        array([[[ 0.55992296,  2.2784546 ,  2.16861887],
                [ 1.41105244, -0.04838148, -0.02537328],
                [-0.68192959, -1.79235591, -1.38495084],
                [ 3.63367216,  1.13841536,  0.925656  ]],
        <BLANKLINE>
               [[-0.36735335, -0.71750135,  1.55387818],
                [ 2.06780495,  0.74576665, -0.87182718],
                [ 0.56217067,  2.4415253 ,  0.37817106],
                [ 1.07437016, -0.31895032,  3.34089204]]])
    """

    def __init__(self, sampler, gamma=1, xi=1, delta=2, lam=2):
        """
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            gamma (np.ndarray): gamma
            xi (np.ndarray): xi
            delta (np.ndarray): delta > 0
            lam (np.ndarray): lambda > 0
        """
        self.parameters = ['gamma', 'xi', 'delta', 'lam']
        self.domain = np.array([[0,1]])
        self.range = np.array([[-np.inf,np.inf]])
        self._parse_sampler(sampler)
        self.gamma = gamma
        self.xi = xi
        self.delta = delta
        self.lam = lam
        if np.isscalar(self.gamma):
            gamma = np.tile(self.gamma,self.d)
        if np.isscalar(self.xi):
            xi = np.tile(self.xi,self.d)
        if np.isscalar(self.delta):
            delta = np.tile(self.delta,self.d)
        if np.isscalar(self.lam):
            lam = np.tile(self.lam,self.d)
        self._gamma = np.array(gamma)
        self._xi = np.array(xi)
        self._delta = np.array(delta)
        self._lam = np.array(lam)
        if len(self._gamma)!=self.d or len(self._xi)!=self.d or len(self._delta)!=self.d or len(self._lam)!=self.d:
            raise DimensionError("all Johnson's S_U parameters be scalar or have length equal to dimension.")
        if (self._delta<=0).any() or (self._lam<=0).any():
            raise ParameterError("delta and lam (lambda) must be greater than 0")
        super(JohnsonsSU,self).__init__()
        assert self._gamma.shape==(self.d,) and self._xi.shape==(self.d,) and self._delta.shape==(self.d,) and self._lam.shape==(self.d,)

    def _transform(self, x):
        return self._lam*np.sinh((norm.ppf(x)-self._gamma)/self._delta)+self._xi
    
    def _weight(self, x):
        term1 = (x-self._xi)/self._lam
        term2 = self._delta/(self._lam*np.sqrt(2*np.pi)) * 1/np.sqrt(1+term1**2)
        term3 = np.exp(-1/2*(self._gamma+self._delta*np.arcsinh(term1))**2)
        return np.prod(term2*term3,-1)
    
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
