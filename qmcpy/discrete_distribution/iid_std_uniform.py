from ._discrete_distribution import IID
from numpy import *


class IIDStdUniform(IID):
    """
    A wrapper around NumPy's IID Standard Uniform generator `numpy.random.rand`.

    >>> dd = IIDStdUniform(dimension=2,seed=7)
    >>> dd.gen_samples(4)
    array([[0.62509547, 0.8972138 ],
           [0.77568569, 0.22520719],
           [0.30016628, 0.87355345],
           [0.0052653 , 0.82122842]])
    >>> dd
    IIDStdUniform (DiscreteDistribution Object)
        d               2^(1)
        entropy         7
        spawn_key       ()
    """

    def __init__(self, dimension=1, seed=None, replications=1):
        """
        Args:
            dimension (int): dimension of samples
            seed (None or int or numpy.random.SeedSeq): seed the random number generator for reproducibility
        """
        self.mimics = 'StdUniform'
        self.low_discrepancy = False
        self.d_max = inf
        self.replications = replications
        super(IIDStdUniform,self).__init__(dimension,seed)

    def gen_samples(self, n):
        """
        Generate samples 

        Args:
            n (int): Number of observations to generate

        Returns:
            ndarray: n x self.d array of samples
        """
        x = self.rng.uniform(size=(self.replications,int(n),self.d))
        return x[0] if self.replications==1 else x
    
    def pdf(self, x):
        return ones(x.shape[0], dtype=float)
        
    def _spawn(self, child_seed, dimension):
        return IIDStdUniform(dimension=dimension,seed=child_seed)
        
