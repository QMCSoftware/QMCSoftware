from ._discrete_distribution import DiscreteDistribution
from numpy import *
from scipy.stats import norm


class IIDStdGaussian(DiscreteDistribution):
    """
    A wrapper around NumPy's IID Standard Uniform generator `numpy.random.randn`.

    >>> dd = IIDStdGaussian(dimension=2,seed=7)
    >>> dd.gen_samples(4)
    array([[ 1.691e+00, -4.659e-01],
           [ 3.282e-02,  4.075e-01],
           [-7.889e-01,  2.066e-03],
           [-8.904e-04, -1.755e+00]])
    >>> dd
    IIDStdGaussian (DiscreteDistribution Object)
        d               2^(1)
        seed            7
        mimics          StdGaussian
    """

    def __init__(self, dimension=1, seed=None):
        """
        Args:
            dimension (int): dimension of samples
            seed (int): seed the random number generator for reproducibility
        """
        self.parameters = ['d','seed','mimics']
        self.d = dimension
        self.set_seed(seed)
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
        return self.rng.randn(int(n), int(self.d))
    
    def pdf(self, x):
        return norm.pdf(x).prod(1)
    
    def set_seed(self,seed):
        self.seed = seed
        self.rng = random.RandomState(self.seed)
    
    def _set_dimension(self, dimension):
        self.d = dimension
        