from .gaussian import Gaussian
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ._true_measure import TrueMeasure
from ..discrete_distribution.lattice.lattice import Lattice
from ..util import DimensionError, ParameterError
from numpy import *
from scipy.special import kv, gamma

class Matern(Gaussian):
    """
    A normal measure using a Matern kernel as the covariance matrix.
    >>> mean = full(31, 1.1)
    >>> x_values = arange(31.0) / 30.0 #[0, 1/30, ..., 1]
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
    array([[1.22067398, 1.21845902, 1.21476923, 1.21929383, 1.23145468,
            1.23591489, 1.23739288, 1.23401112, 1.23056609, 1.22649471,
            1.2199999 , 1.21871583, 1.22854283, 1.24212726, 1.25115155,
            1.2558769 , 1.26196885, 1.2728588 , 1.2737948 , 1.27271341,
            1.27067302, 1.25991003, 1.24649103, 1.24080074, 1.2430361 ,
            1.24500147, 1.24922986, 1.24655733, 1.24378233, 1.23947814,
            1.23729738]])
    >>> x_values = array([0, 1, 2])
    >>> mean = full(3, 1.1)
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
            points (ndarray): The positions of points on a metric space. The array
                should be of shape n x d: n points, each of dimension d.
            length_scale (float): Determines "peakiness", or how correlated 
                two points are based on their distance.
            nu (float): The "smoothness" of the Matern function. e.g. nu=1.5
                implies a once-differentiable function, while nu=2.5 implies twice
                differentiability. Meanwhile, when nu=0.5, the Matern kernel equals
                the squared exponential kernel, while as nu approaches inf, it equals 
                the RBF kernel. Note that nu values not in [0.5, 1.5, 2.5, inf] 
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
        if not (isinstance(points, ndarray)):
            raise ParameterError("Must pass in a points ndarray.")
        if not (sampler.d == len(mean) and points.shape[0] == len(mean)):
            raise DimensionError("The lengths of the sampler and mean array and the number of points must all be equal.")
        
        if len(points.shape) == 1: #one dimensional points array, 1 x N
            points = atleast_2d(points).T
        self.points = points
        self.length_scale = length_scale
        self.nu = nu
        self.variance = variance
        
        # See [1], [2] for Matern formula
        # Replicating scikit-learn's Matern kernel using hard-coded cases, scipy.special.kv
        dimension = len(mean)
        rho = length_scale
        covariance = zeros((dimension, dimension))
        
        for i in range(dimension):
            for j in range(dimension):
                d = linalg.norm(points[i] - points[j])
                if d == 0:
                    covariance[i][j] = 1 #convergent value
                elif isclose(nu, 0.5):
                    covariance[i][j] = exp(-1 * d / rho)
                elif isclose(nu, 1.5):
                    covariance[i][j] = (1 + sqrt(3) * d / rho) * exp(-1 * sqrt(3) * d / rho)
                elif isclose(nu, 2.5):
                    covariance[i][j] = ((1 + sqrt(5) * d / rho + 5 * d ** 2 / (3 * rho ** 2)) * 
                                        exp(-1 * sqrt(5) * d / rho))
                else:
                    k = sqrt(2 * nu) * d / rho
                    covariance[i][j] = 2 ** (1 - nu) / gamma(nu) * k ** nu * kv(nu, k)
                """ this doesn't produce a positive definite matrix, so it isn't accepted in Gaussian
                elif nu == inf:
                    covariance[i][j] = exp(-1 * d ** 2 / (2 * rho ** 2))
                """
                covariance[i][j] = covariance[i][j] * variance
        
        #print(covariance)
        super().__init__(sampler, mean=mean, covariance=covariance, decomp_type=decomp_type)

    def _spawn(self, sampler):
        return Matern(sampler, self.points, length_scale=self.length_scale, nu=self.nu, variance=self.variance, 
                      mean=self.mean, decomp_type=self.decomp_type)
        
