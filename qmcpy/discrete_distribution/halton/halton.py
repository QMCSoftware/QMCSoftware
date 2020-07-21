from .._discrete_distribution import DiscreteDistribution
from ..qrng import ghalton_qrng
from .owen_halton import rhalton
from ...util import ParameterWarning, ParameterError
import warnings
from numpy import random


class Halton(DiscreteDistribution):
    """
    Quasi-Random Generalize Halton nets.

    >>> h = Halton(2,seed=7)
    >>> h.gen_samples(1)
    array([[0.315, 0.739]])
    >>> h.gen_samples(1)
    array([[0.315, 0.739]])
    >>> h.set_dimension(4)
    >>> h.set_seed(8)
    >>> h.gen_samples(2)
    array([[0.276, 0.382, 0.502, 0.356],
           [0.776, 0.049, 0.702, 0.784]])
    >>> h
    Halton (DiscreteDistribution Object)
        dimension       2^(2)
        generalize      1
        seed            2^(3)
        backend         owen
        mimics          StdUniform
    
    References
        Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.

        Owen, A. B. "A randomized Halton algorithm in R," 2017. arXiv:1706.02808 [stat.CO]
    """

    parameters = ['dimension','generalize','seed','backend','mimics']

    def __init__(self, dimension=1, generalize=True, randomize=True, backend='Owen', seed=None):
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
            Halton instances are automatically randomized. 
            See 'QRNG' and 'Owen' backend references for specific randomization methods and differences. 
        """
        self.dimension = dimension
        self.generalize = generalize
        self.backend = backend.lower()
        self.mimics = 'StdUniform'
        self.seed = seed
        self.randomize = randomize
        self.low_discrepancy = True
        self.set_seed(self.seed)
        if self.backend not in ['qrng','owen']:
            raise ParameterError('Halton backend must be either "QRNG" or "Owen"')
        if self.backend=='owen' and self.generalize==False:
            warnings.warn('\nHalton with "Owen" backend must have generalize=True. '+\
                'Using "QRNG" backend',ParameterWarning)
            self.backend='qrng'
        if self.backend=='qrng' and self.randomize==False:
            warnings.warn('\nHalton with "QRNG" backend must be randomized. ' +\
                'Using "Owen" backend with randomize=False', ParameterWarning);
            self.backend='owen'
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
        if (not n) and (n_min!=0) and self.backend=='qrng':
            raise ParameterError('\nHalton with "QRNG" backend does not support skipping samples. '+\
                'Use "Owen" backend in order to provide n_min!=0')
        if n_min == 0 and self.randomize==False and warn:
            warnings.warn("Non-randomized Halton sequence includes the origin",ParameterWarning)
        n = int(n_max-n_min)
        d = int(self.dimension)
        if self.backend=='qrng':
            x = ghalton_qrng(n,d,self.generalize,self.seed)
        if self.backend=='owen':
            x = rhalton(n, d, n0=int(n_min), d0=0, randomize=self.randomize,singleseed=self.seed)
        return x

    def set_seed(self, seed):
        """ See abstract method. """
        self.seed = seed if seed else random.randint(2**32)
        
    def set_dimension(self, dimension):
        """ See abstract method. """
        self.dimension = dimension