from ._discrete_distribution import DiscreteDistribution
from numpy import *
from scipy.stats import norm


class IIDStdGaussian(DiscreteDistribution):
    """
    A wrapper around NumPy's IID Standard Uniform generator `numpy.random.randn`.

    >>> dd = IIDStdGaussian(dimension=2,seed=7)
    >>> dd.gen_samples(4)
    >>> dd
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

    def _spawn(self, s, child_seeds, dimensions):
        return [IIDStdGaussian(dimension=dimensions[i],seed=child_seeds[i]) for i in range(s)]
        
        