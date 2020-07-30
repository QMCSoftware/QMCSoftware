from ._discrete_distribution import DiscreteDistribution
from numpy import random

class CustomIIDDistribution(DiscreteDistribution):
    """
    Custom IID Discrte Distribution.

    >>> random.seed(7)
    >>> cd = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,3)))
    >>> cd
    CustomIIDDistribution (DiscreteDistribution Object)
        dimension       None
    >>> cd.gen_samples(2)
    array([[6, 3, 3],
           [4, 6, 6]])
    """
    parameters = ['dimension']

    def __init__(self, custom_generator):
        """
        Args:
            custom_generator (function): custom generator of discrete distribution
        """
        self.custom_generator = custom_generator
        self.distrib_type = 'iid'
        self.mimics = 'Custom'
        self.dimension = None
        self.low_discrepancy = False
        super(CustomIIDDistribution,self).__init__()

    def gen_samples(self, n):
        """
        Generate samples 

        Args:
            n (int): Number of observations to generate

        Returns:
            ndarray: n x d (dimension) array of samples
        """
        return self.custom_generator(n)

