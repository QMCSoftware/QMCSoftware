"""
Definitions for IIDDistribution, a DiscreteDistribution
"""
from numpy import random

from . import DiscreteDistribution

class IIDDistribution(DiscreteDistribution):
    """
    Specifies and generates the components of :math:`\dfrac{1}{n} \sum_{i=1}^n \delta_{\mathbf{x}_i}(\cdot)`
    where the :math:`\mathbf{x}_i` are IIDDistribution uniform on :math:`[0,1]^d` or IIDDistribution standard Gaussian
    """
    
    def __init__(self, true_distrib=None, seed_rng=None):
        """        
        Args:
            accepted_measures (list of strings): Measure objects compatible with the DiscreteDistribution
            seed_rng (int): seed for random number generator to ensure reproduciblity            
        """
        accepted_measures = ['StdUniform','StdGaussian'] # IID Distribution generators
        if seed_rng: random.seed(seed_rng) # numpy.random for underlying generation
        super().__init__(accepted_measures, true_distrib)

    def gen_distrib(self, n, m):
        """
        Generate j nxm samples from the true-distribution
        
        Args:       
            n (int): Number of observations (sample.size()[1])
            m (int): Number of dimensions (sample.size()[2])        
        Returns:
            nxm (numpy array) 
        """
        if type(self.true_distrib).__name__=='StdUniform': # IID U(0,1)
            return random.rand(int(n),int(m)) # nxm
        elif type(self.true_distrib).__name__=='StdGaussian': # IID N(0,1)
            return random.randn(int(n),int(m)) # nxm