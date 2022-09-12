from .._discrete_distribution import DiscreteDistribution
from ...util import ParameterError, ParameterWarning
from numpy.core.numeric import isscalar
from numpy import *
import warnings

class SobolSciPy(DiscreteDistribution):
    """
    See https://scipy.github.io/devdocs/reference/generated/scipy.stats.qmc.Sobol.html
    
    >>> s = SobolSciPy(2,seed=7)
    >>> s.gen_samples(4)
    array([[5.79259991e-01, 7.40284680e-01],
           [4.15829644e-02, 6.92069530e-04],
           [4.78844851e-01, 7.75258362e-01],
           [8.92499685e-01, 4.83783960e-01]])
    >>> s.gen_samples(1)
    array([[0.57925999, 0.74028468]])
    >>> s.gen_samples(1)
    array([[0.57925999, 0.74028468]])
    >>> s
    SobolSciPy (DiscreteDistribution Object)
        d               2^(1)
        randomize       1
        graycode        1
        entropy         7
        spawn_key       ()
    >>> SobolSciPy(dimension=2,randomize=False).gen_samples(n_min=2,n_max=4)
    array([[0.75, 0.25],
           [0.25, 0.75]])
    >>> sobolscipy = SobolSciPy(dimension=2,randomize=False)
    >>> sobolscipy.gen_samples(n=4,warn=False)
    array([[0.  , 0.  ],
           [0.5 , 0.5 ],
           [0.75, 0.25],
           [0.25, 0.75]])
    >>> sobolscipy.gen_samples(n_min=4,n_max=8,warn=False) # only uses scipy.stats.qmc.Sobol.random()
    array([[0.375, 0.375],
           [0.875, 0.875],
           [0.625, 0.125],
           [0.125, 0.625]])
    >>> sobolscipy.gen_samples(n_min=2, n_max=4,warn=False) # uses scipy.stats.qmc.Sobol.reset() & ...fast_forward()
    array([[0.75, 0.25],
           [0.25, 0.75]])
    >>> sobolscipy.gen_samples(n_min=6,n_max=8,warn=False) # uses scipy.stats.qmc.Sobol.fast_forward()
    array([[0.625, 0.125],
           [0.125, 0.625]])
    """

    def __init__(self, dimension=1, randomize=True, seed=None):
        """
        Args:
            dimension (int): dimension of samples
            randomize (bool): scramble points? 
            seeds (list): int seed of list of seeds, one for each dimension.
        """
        try:
            from scipy.stats.qmc import Sobol as SobolScipyOG
        except:
            raise ParameterError("scipy.stats.qmc.Sobol not found, try updating to scipy>=1.7.0")
        self.parameters = ['randomize','graycode']
        if not isscalar(dimension):
            raise ParameterError("SobolSciPy does not support vectorized dimension indexing")
        self.randomize = randomize
        self.graycode = True
        self.mimics = 'StdUniform'
        self.low_discrepancy = True
        self.d_max = 21201
        super(SobolSciPy,self).__init__(dimension,seed)        
        self.sobol = SobolScipyOG(self.d,scramble=self.randomize,seed=self.entropy)
        self.ncurrent = 0

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True):
        """
        Generate samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples. 
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.

        Returns:
            ndarray: (n_max-n_min) x d (dimension) array of samples
        """
        if n:
            n_min = 0
            n_max = n
        if n_min == 0 and self.randomize==False and warn:
            warnings.warn("Non-randomized Sobol sequence includes the origin",ParameterWarning)
        n = int(n_max-n_min)
        to_skip_ahead = n_min-self.ncurrent
        self.ncurrent = n_max
        if to_skip_ahead<0:
            self.sobol.reset()
            if n_min>0:
                self.sobol.fast_forward(n_min)
        elif to_skip_ahead>0:
            self.sobol.fast_forward(to_skip_ahead)
        x = self.sobol.random(n)
        return x.reshape(-1,self.d)
    
    def pdf(self, x):
        """ pdf of a standard uniform """
        return ones(x.shape[0], dtype=float)
