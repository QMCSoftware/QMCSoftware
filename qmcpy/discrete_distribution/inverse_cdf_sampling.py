from ._discrete_distribution import DiscreteDistribution
from . import Sobol
from ..util import TransformError
from numpy import *


class InverseCDFSampling(DiscreteDistribution):
    """
    Sampling by inverse CDF transform applied to 
    discrete distribution samples mimics standard uniform. 

    >>> _lambda = 1.5
    >>> exp_pdf = lambda x,l=_lambda: l*exp(-l*x)
    >>> exp_inverse_cdf = lambda u,l=_lambda: -log(1-u)/l
    >>> exponential_measure = InverseCDFSampling(
    ...     distribution_mimicking_uniform = Sobol(dimension=2,seed=7),
    ...     inverse_cdf_fun = exp_inverse_cdf)
    >>> exponential_measure
    InverseCDFSampling (DiscreteDistribution Object)
        dimension       2^(1)
    >>> exponential_measure.gen_samples(n_min=4,n_max=8)
    array([[1.296, 0.199],
           [0.294, 0.947],
           [0.623, 0.474],
           [0.075, 0.006]])
    
    Math for above example:
        - $y \\sim \\text{Exp}(l)$
        - $\\text{pdf }y: f(x) = l*\\exp(-l*x)$
        - $\\text{cdf }y: F(x)= 1-\\exp(-l*x)$
        - $F^{-1}(u) = -\\log(1-u)/l \\sim \\text{Exp}(l) \\text{ for } u \\sim \\text{Uniform}(0,1)$
    """

    parameters = ['dimension']

    def __init__(self, distribution_mimicking_uniform, inverse_cdf_fun=lambda u: u):
        """
        Args:
            distribution_mimicking_uniform (DiscreteDistribution): DiscreteDistribution instance 
                which mimics standard uniform
            inverse_cdf_fun (function): function accepting samples mimicing uniform 
                and applying inverse CDF transform. Must have 1 input argument accepting an  
                ndarray of size n (observations) by d (diemsion) 
        """
        self.distribution_u = distribution_mimicking_uniform
        self.inverse_cdf_fun = inverse_cdf_fun
        if self.distribution_u.mimics != 'StdUniform':
            raise TransformError(\
                'Can only apply inverse CDF transform to DiscreteDistributions mimicing StdUniform')
        self.low_discrepancy = True if 'IID' in type(self.distribution_u).__name__ else False
        self.dimension = self.distribution_u.dimension
        self.mimics = 'None'
        super(InverseCDFSampling,self).__init__()
    
    def gen_samples(self, *args, **kwargs):
        """
        Generate samples 

        Args:
            n (int): Number of observations to generate

        Returns:
            ndarray: n x d (dimension) array of samples
        """
        original_samples = self.distribution_u.gen_samples(*args, **kwargs)
        mimic_samples = self.inverse_cdf_fun(original_samples)
        return mimic_samples
