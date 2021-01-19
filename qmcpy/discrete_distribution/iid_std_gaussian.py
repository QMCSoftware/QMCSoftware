from ._discrete_distribution import DiscreteDistribution
from numpy import *
from scipy.stats import norm


class IIDStdGaussian(DiscreteDistribution):
    """
    A wrapper around NumPy's IID Standard Uniform generator `numpy.random.randn`.

    >>> dd = IIDStdGaussian(dimension=2,seed=7)
    >>> dd.gen_samples(4)
    >>> dd
    IIDStdGaussian (DiscreteDistribution Object)
        d               2^(1)
        seed            7
        mimics          StdGaussian
    """

    parameters = ['d','seed','mimics']

    def __init__(self, dimension=1, seed=None):
        """
        Args:
            dimension (int): dimension of samples
            seed (int): seed the random number generator for reproducibility
        """
        self.d = dimension
        self.seed = seed
        random.seed(self.seed)
        self.mimics = 'StdGaussian'
        self.low_discrepancy = False
        super(IIDStdGaussian,self).__init__()

    def gen_samples(self, n):
        """
        Generate samples 

        Args:
            n (int): Number of observations to generate

        Returns:
            ndarray: n x self.d array of samples
        """
        return random.randn(int(n), int(self.d))
    
    def pdf(self, x):
        return norm.pdf(x).prod(1)
    
    def _set_dimension(self, dimension):
        self.d = dimension
        