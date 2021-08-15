from ._discrete_distribution import DiscreteDistribution
from numpy import *
from scipy.stats import norm


class IIDStdGaussian(DiscreteDistribution):
    """
    A wrapper around NumPy's IID Standard Uniform generator `numpy.random.randn`.

    >>> dd = IIDStdGaussian(dimension=2,seed=7)
    >>> dd.gen_samples(4)
    array([[ 0.76412904,  1.38253623],
           [ 1.59603668, -0.34658357],
           [ 0.75751054, -1.75095847],
           [ 0.14504699,  0.0736317 ]])
    >>> dd
    IIDStdGaussian (DiscreteDistribution Object)
        d               2^(1)
        entropy         7
        spawn_key       ()
    """

    def __init__(self, dimension=1, seed=None):
        """
        Args:
            dimension (int): dimension of samples
            seed (None or int or numpy.random.SeedSeq): seed the random number generator for reproducibility
        """
        self.mimics = 'StdGaussian'
        self.low_discrepancy = False
        self.d_max = inf
        super(IIDStdGaussian,self).__init__(dimension,seed)

    def gen_samples(self, n):
        """
        Generate samples 

        Args:
            n (int): Number of observations to generate

        Returns:
            ndarray: n x self.d array of samples
        """
        return self.rng.normal(size=(n,self.d))
    
    def pdf(self, x):
        return norm.pdf(x).prod(1)

    def _spawn(self, child_seed, dimension):
        return IIDStdGaussian(dimension=dimension,seed=child_seed)
        
        