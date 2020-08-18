from .._discrete_distribution import DiscreteDistribution
from .halton_owen import HaltonOwen
from .halton_qrng import HaltonQRNG
from ...util import ParameterError
from numpy import random


class Halton(DiscreteDistribution):
    """
    Quasi-Random Halton nets.

    >>> h = Halton(2,seed=7)
    >>> h.gen_samples(1)
    array([[0.218, 0.931]])
    >>> h.gen_samples(1)
    array([[0.218, 0.931]])
    >>> h.set_dimension(4)
    >>> h.set_seed(8)
    >>> h.gen_samples(2)
    array([[0.895, 0.172, 0.048, 0.66 ],
           [0.395, 0.838, 0.448, 0.231]])
    >>> h
    Halton (DiscreteDistribution Object)
        dimension       2^(2)
        generalize      1
        randomize       1
        seed            2^(3)
        mimics          StdUniform
        backend         OWEN
    
    References:

        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.
        
        [2] Owen, A. B. "A randomized Halton algorithm in R," 2017. arXiv:1706.02808 [stat.CO]
    """

    parameters = ['dimension','generalize','randomize','seed','mimics','backend']

    def __init__(self, dimension=1, generalize=True, randomize=True, seed=None, backend='Owen'):
        """
        Args:
            dimension (int): dimension of samples
            generalize (bool): generalize the Halton sequence?
            randomize (bool): randomize the sequence
            backend (str): Backend generator. Must be "QRNG" or "Owen". 
                "QRNG" backend supports both plain and generalized Halton with randomization and generally provides better accuracy. 
                "Owen" only supports generalized Halton but allows for n_min!=0 in 'gen_samples' method.
            seed (int): seed the random number generator for reproducibility
        
        Note:
            See References [1] and [2] for specific randomization methods and differences. 
        """
        self.backend = backend.upper()
        backend_objs = {'OWEN':HaltonOwen,'QRNG':HaltonQRNG}
        backends = list(backend_objs.keys())
        if self.backend not in backends:
            raise ParameterError('Halton requires backend be in %s'%(str(backends)))
        self.generator = backend_objs[self.backend](dimension,generalize,randomize,seed)
        self.dimension, self.generalize, self.randomize, self.seed = self.generator.get_params()
        self.low_discrepancy = True
        self.mimics = 'StdUniform'
        super(Halton,self).__init__()

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
        x = self.generator.gen_samples(n_min,n_max,warn)
        return x

    def set_seed(self, seed):
        """ See abstract method. """
        self.seed = self.generator.set_seed(seed)
        
    def set_dimension(self, dimension):
        """ See abstract method. """
        self.dimension = self.generator.set_dimension(dimension)
