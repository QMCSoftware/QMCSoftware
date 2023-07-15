from .gaussian import Gaussian
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ._true_measure import TrueMeasure
from ..discrete_distribution.lattice.lattice import Lattice
from ..util import DimensionError, ParameterError
import numpy as np
from scipy.stats import norm, multivariate_normal
from sklearn import gaussian_process as gp

class Matern(Gaussian):
    """
    A normal measure using a Matern kernel as the covariance matrix.
    >>> import numpy as np
    >>> mean = np.full(31, 1.1)
    >>> x_values = np.arange(31.0) / 30.0  # [0, 1/30, ..., 1]
    >>> m = Matern(Lattice(dimension=9, seed=7), x_values, length_scale = 0.5, variance = 0.01, mean=mean)
    >>> m
    Matern (TrueMeasure Object)
    mean            [1.1 1.1 1.1 ... 1.1 1.1 1.1]
    covariance      [[0.01  0.01  0.01  ... 0.002 0.002 0.001]
                    [0.01  0.01  0.01  ... 0.002 0.002 0.002]
                    [0.01  0.01  0.01  ... 0.002 0.002 0.002]
                    ...
                    [0.002 0.002 0.002 ... 0.01  0.01  0.01 ]
                    [0.002 0.002 0.002 ... 0.01  0.01  0.01 ]
                    [0.001 0.002 0.002 ... 0.01  0.01  0.01 ]]
    decomp_type     PCA
    >>> m.gen_samples(1)
    array([[1.22948461, 1.22690105, 1.22726142, 1.2194079 , 1.21299668,
        1.20616204, 1.20012509, 1.19676366, 1.20517634, 1.21533291,
        1.22050969, 1.21863876, 1.21207729, 1.20128901, 1.18697989,
        1.17185631, 1.15863686, 1.14607381, 1.12794139, 1.11301057,
        1.10090007, 1.07881531, 1.05168982, 1.02405704, 0.99693782,
        0.98095279, 0.96809501, 0.96005229, 0.9515122 , 0.94952871,
        0.95042189]])
    """
    def __init__(self, sampler, x_values, length_scale = 1.0, nu = 1.5, variance = 1.0, mean = [], decomp_type='PCA'):
        """
        Matern kernel: calculates covariance over a metric space based only on the distance between points.
        More information can be found at 
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html
        
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform. 
            x_values (array): The positions of points on a metric space.
            length_scale (float): Determines "peakiness", or how correlated 
                two points are based on their distance.
            nu (float): The "smoothness" of the Matern function. e.g. nu=1.5
                implies a once-differentiable function, while nu=2.5 implies twice
                differentiability. Meanwhile, when nu=0.5, the Matern kernel equals
                the RBF kernel, while when nu=inf, it equals the absolute exponential
                kernel. Note that nu values not in [0.5, 1.5, 2.5, inf] will be 
                ~10x slower to run.
            variance (float): The variance (or the diagonal elements of the
                covariance matrix) at each point. 
            mean (float): mu for Normal(mu,sigma^2)
            decomp_type (str): method of decomposition either  
                "PCA" for principal component analysis or 
                "Cholesky" for cholesky decomposition.
        """
        #assert len(x_values) == len(mean)
        if not (len(x_values) == len(mean)):
            raise DimensionError("The dimensions of the position array and means must be equal.")
        if not (isinstance(sampler, DiscreteDistribution) or isinstance(sampler, TrueMeasure)):
            raise ParameterError("sampler input should either be a DiscreteDistribution or TrueMeasure.")
        self.x_values = x_values
        self.length_scale = length_scale
        self.nu = nu
        self.variance = variance
        kernel = gp.kernels.Matern(length_scale = length_scale, nu=nu)
        tnp = np.array([x_values]).T
        #print(tnp)
        self.covariance = variance * kernel.__call__(tnp) #takes array of size (# samples) x (# data points)
        #print(covariances)
        sampler_out = sampler.spawn(1, len(x_values))[0] #new version of sampler
        super().__init__(sampler_out, mean=mean, covariance=self.covariance, decomp_type=decomp_type)

    def _spawn(self, sampler):
        return Matern(sampler.spawn(1, len(self.x_values))[0], self.x_values, length_scale=self.length_scale, nu=self.nu, variance=self.variance, 
                      mean=self.mean, decomp_type=self.decomp_type)
        

