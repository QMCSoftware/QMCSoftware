from ._true_measure import TrueMeasure
from .uniform import Uniform
from .gaussian import Gaussian
from ..discrete_distribution import Sobol
from ..util import TransformError, ParameterError
from scipy.stats import norm
from numpy import *


class Lebesgue(TrueMeasure):
    """
    >>> d = 2
    >>> dd = Sobol(d,seed=7)
    >>> Lebesgue(transformer=Uniform(d,lower_bound=[-1,0],upper_bound=[1,3]))
    Lebesgue (TrueMeasure Object)
        transformer     Uniform (TrueMeasure Object)
                           d               2^(1)
                           lower_bound     [-1  0]
                           upper_bound     [1 3]
    >>> lg = Lebesgue(transformer=Gaussian(d,mean=[0,0],covariance=[[1,0],[0,1]]))
    >>> lg
    Lebesgue (TrueMeasure Object)
        transformer     Gaussian (TrueMeasure Object)
                           d               2^(1)
                           mean            [0 0]
                           covariance      [[1 0]
                                           [0 1]]
                           decomp_type     pca
    >>> lg.set_transform(Gaussian(d,mean=[0,1],covariance=[[2,0],[0,2]]))
    >>> lg
    Lebesgue (TrueMeasure Object)
        transformer     Gaussian (TrueMeasure Object)
                           d               2^(1)
                           mean            [0 1]
                           covariance      [[2 0]
                                           [0 2]]
                           decomp_type     pca
    """

    parameters = ['transformer']
    
    def __init__(self, transformer):
        """
        Args:
            transformer (TrueMeasure): A measure whose transform will be used.
        """
        self.transformer = transformer
        self.d = self.transformer.d
        self.set_transform(self.transformer)
        self.set_dimension = self.transformer.set_dimension
        super(Lebesgue,self).__init__()

    def weight(self, x):
        return ones(x.shape[0],dtype=float)

    def set_dimension(self, dimension):
        self.measure.set_dimension(dimension)
