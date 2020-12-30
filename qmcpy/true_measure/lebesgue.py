from ._true_measure import TrueMeasure
from ..discrete_distribution import Sobol
from ..util import TransformError, ParameterError
from numpy import array, isfinite, inf, isscalar, tile
from scipy.stats import norm


class Lebesgue(TrueMeasure):
    """
    >>> d = 2
    >>> dd = Sobol(d,seed=7)
    >>> Lebesgue(transform=Uniform(lower_bound=[-1,0],upper_bound=[1,3]))
    Lebesgue (TrueMeasure Object)
        transform = Gaussian(
                        lower_bound     [-1  0]
                        upper_bound     [1 3])
    >>> Lebesgue(transform=Gaussian(mean=[0,0]),covariance=[[1,0],[0,1]]))
    Lebesgue (TrueMeasure Object)
        lower_bound     [-inf -inf]
        upper_bound     [inf inf]
    """

    parameters = ['transform']
    
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
        return ones(x.shape[0],dtye=float)

    def set_dimension(self, dimension):
        self.measure.set_dimension(dimension)
