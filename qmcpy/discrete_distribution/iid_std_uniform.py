from ._discrete_distribution import DiscreteDistribution
from numpy import *


class IIDStdUniform(DiscreteDistribution):
    """
    A wrapper around NumPy's IID Standard Uniform generator `numpy.random.rand`.

    >>> dd = IIDStdUniform(dimension=2,seed=7)
    >>> dd.gen_samples(4)
    array([[0.04386058, 0.58727432],
           [0.3691824 , 0.65212985],
           [0.69669968, 0.10605352],
           [0.63025643, 0.13630282]])
    >>> dd
    IIDStdUniform (DiscreteDistribution Object)
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
        self.mimics = 'StdUniform'
        self.low_discrepancy = False
        self.d_max = inf
        super(IIDStdUniform,self).__init__(dimension,seed)

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
        