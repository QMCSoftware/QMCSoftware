from .gaussian import Gaussian
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ._true_measure import TrueMeasure
from ..discrete_distribution.lattice.lattice import Lattice
from ..util import DimensionError, ParameterError
import numpy as np
from scipy.special import kv, gamma

class Matern(Gaussian):
    """
    A normal measure using a Matern kernel as the covariance matrix.
    >>> mean = np.full(31, 1.1)
    >>> x_values = np.arange(31.0) / 30.0 #[0, 1/30, ..., 1]
    >>> m = Matern(Lattice(dimension=31, seed=7), x_values, length_scale = 0.5, variance = 0.01, mean=mean)
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
    array([[1.20120653, 1.20567888, 1.21279218, 1.22049411, 1.21721125,
            1.21153221, 1.2175453 , 1.21161981, 1.19718766, 1.18234845,
            1.16702627, 1.15746868, 1.1467863 , 1.13942278, 1.13055293,
            1.11774413, 1.10046122, 1.07650658, 1.05805753, 1.0481866 ,
            1.0380732 , 1.03773682, 1.05539754, 1.06971354, 1.07991351,
            1.09075867, 1.09452842, 1.09035532, 1.07834649, 1.06645406,
            1.05466388]])
    >>> x_values = np.array([0, 1, 2])
    >>> mean = np.full(3, 1.1)
    >>> m3 = Matern(Lattice(dimension = 3,seed=7), x_values, length_scale = 0.5, nu = 3.5, variance = 0.01, mean=mean, decomp_type = 'Cholesky')
    >>> m3
    Matern (TrueMeasure Object)
        mean            [1.1 1.1 1.1]
        covariance      [[1.000e-02 1.378e-03 3.432e-05]
                        [1.378e-03 1.000e-02 1.378e-03]
                        [3.432e-05 1.378e-03 1.000e-02]]
        decomp_type     CHOLESKY

     References:

        [1] scikit-learn developers, Matern kernel. 
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html.
        Accessed 2023.

        [2] Abramowitz and Stegun (1972). Handbook of Mathematical Functions with Formulas, Graphs, 
        and Mathematical Tables. ISBN 0-486-61272-4. (Accessed through Wikipedia, 
        https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function.)
    """

    def __init__(self, sampler, points, length_scale = 1.0, nu = 1.5, variance = 1.0, mean = [], decomp_type='PCA'):
        """
        Matern kernel: calculates covariance over a metric space based only on the distance between points.
        More information can be found at [1].
        
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform. 
            points (np.ndarray): The positions of points on a metric space. The array
                should be of shape n x d: n points, each of dimension d.
            length_scale (float): Determines "peakiness", or how correlated 
                two points are based on their distance.
            nu (float): The "smoothness" of the Matern function. e.g. nu=1.5
                implies a once-differentiable function, while nu=2.5 implies twice
                differentiability. Meanwhile, when nu=0.5, the Matern kernel equals
                the squared exponential kernel, while as nu approaches np.inf, it equals 
                the RBF kernel. Note that nu values not in [0.5, 1.5, 2.5, np.inf] 
                will be ~10x slower to run.
            variance (float): The variance (or the diagonal elements of the
                covariance matrix) of the distribution at each point. Implemented
                as a scaling factor of the Matern covariance. The default 
                value is 1.0.
            mean (float): mu for Normal(mu,sigma^2)
            decomp_type (str): method of decomposition either  
                "PCA" for principal component analysis or 
                "Cholesky" for cholesky decomposition.
        """
        if not (isinstance(sampler, DiscreteDistribution) or isinstance(sampler, TrueMeasure)):
            raise ParameterError("sampler input should either be a DiscreteDistribution or TrueMeasure.")
        if not (isinstance(points, np.ndarray)):
            raise ParameterError("Must pass in a points np.ndarray.")
        if not (sampler.d == len(mean) and points.shape[0] == len(mean)):
            raise DimensionError("The lengths of the sampler and mean array and the number of points must all be equal.")
        
        if len(points.shape) == 1: #one dimensional points array, 1 x N
            points = np.atleast_2d(points).T
        self.points = points
        self.length_scale = length_scale
        self.nu = nu
        self.variance = variance
        
        # See [1], [2] for Matern formula
        # Replicating scikit-learn's Matern kernel using hard-coded cases, scipy.special.kv
        dimension = len(mean)
        rho = length_scale
        covariance = np.zeros((dimension, dimension))
        
        for i in range(dimension):
            for j in range(dimension):
                d = np.linalg.norm(points[i] - points[j])
                if d == 0:
                    covariance[i][j] = 1 #convergent value
                elif np.isclose(nu, 0.5):
                    covariance[i][j] = np.exp(-1 * d / rho)
                elif np.isclose(nu, 1.5):
                    covariance[i][j] = (1 + np.sqrt(3) * d / rho) * np.exp(-1 * np.sqrt(3) * d / rho)
                elif np.isclose(nu, 2.5):
                    covariance[i][j] = ((1 + np.sqrt(5) * d / rho + 5 * d ** 2 / (3 * rho ** 2)) * 
                                        np.exp(-1 * np.sqrt(5) * d / rho))
                else:
                    k = np.sqrt(2 * nu) * d / rho
                    covariance[i][j] = 2 ** (1 - nu) / gamma(nu) * k ** nu * kv(nu, k)
                """ this doesn't produce a positive definite matrix, so it isn't accepted in Gaussian
                elif nu == np.inf:
                    covariance[i][j] = np.exp(-1 * d ** 2 / (2 * rho ** 2))
                """
                covariance[i][j] = covariance[i][j] * variance
        
        #print(covariance)
        super().__init__(sampler, mean=mean, covariance=covariance, decomp_type=decomp_type)

    def _spawn(self, sampler):
        return Matern(sampler, self.points, length_scale=self.length_scale, nu=self.nu, variance=self.variance, 
                      mean=self.mean, decomp_type=self.decomp_type)
        
