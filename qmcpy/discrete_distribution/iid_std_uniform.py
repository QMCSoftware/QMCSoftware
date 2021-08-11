from ._discrete_distribution import DiscreteDistribution
from numpy import *


class IIDStdUniform(DiscreteDistribution):
    """
    A wrapper around NumPy's IID Standard Uniform generator `numpy.random.rand`.

    >>> dd = IIDStdUniform(dimension=2,seed=7)
    >>> dd.gen_samples(4)
    array([[0.07630829, 0.77991879],
           [0.43840923, 0.72346518],
           [0.97798951, 0.53849587],
           [0.50112046, 0.07205113]])
    >>> dd
    IIDStdUniform (DiscreteDistribution Object)
        d               2^(1)
        seed            7
        mimics          StdUniform
    """

    def __init__(self, dimension=1, seed=None):
        """
        Args:
            dimension (int): dimension of samples
            seed (None or int or numpy.random.SeedSeq): seed the random number generator for reproducibility
        """
        self.parameters = ['d']
        self.d = dimension
        self.mimics = 'StdUniform'
        self.low_discrepancy = False
        super(IIDStdUniform,self).__init__(seed)

    def gen_samples(self, n):
        """
        Generate samples 

        Args:
            n (int): Number of observations to generate

        Returns:
            ndarray: n x self.d array of samples
        """
        return self.rng.uniform(size=(n,self.d))
    
    def pdf(self, x):
        return ones(x.shape[0], dtype=float)
        
    def _spawn(self, s, child_seeds, dimensions):
        return [IIDStdUniform(dimension=dimensions[i],seed=child_seeds[i]) for i in range(s)]
        