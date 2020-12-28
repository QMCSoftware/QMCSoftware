from ._discrete_distribution import DiscreteDistribution
from numpy import random


class IIDStdGaussian(DiscreteDistribution):
    """
    A wrapper around NumPy's IID Standard Gaussian generator `numpy.random.randn`. 

    >>> dd = IIDStdGaussian(dimension=2,seed=7)
    >>> dd
    IIDStdGaussian (DiscreteDistribution Object)
        dimension       2^(1)
        seed            7
        mimics          StdGaussian
    >>> dd.gen_samples(4)
    array([[ 1.691e+00, -4.659e-01],
           [ 3.282e-02,  4.075e-01],
           [-7.889e-01,  2.066e-03],
           [-8.904e-04, -1.755e+00]])
    >>> dd.set_dimension(3)
    >>> x = dd.gen_samples(5)
    >>> x.shape
    (5, 3)
    """
    
    parameters = ['dimension', 'seed', 'mimics']

    def __init__(self, dimension=1, seed=None):
        """
        Args:
            dimension (int): dimension of samples
            seed (int): seed the random number generator for reproducibility
        """
        self.dimension = dimension
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
            ndarray: n x d (dimension) array of samples
        """
        return random.randn(int(n), int(self.dimension))
    
    def set_dimension(self, dimension):
        """ See abstract method. """
        self.dimension = dimension
