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
    >>> x_values = atleast_2d(arange(31.0) / 30.0).T  # [[0], [1/30], ..., [1]]
    >>> m = Matern(Lattice(dimension=31, seed=7), points = x_values, length_scale = 0.5, variance = 0.01, mean=mean)
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
    array([[1.22818861, 1.22050757, 1.21522052, 1.2144905 , 1.21449756,
            1.22499156, 1.24161076, 1.25371086, 1.26118833, 1.2718846 ,
            1.27662595, 1.28237625, 1.28688038, 1.28466947, 1.28328019,
            1.27822864, 1.26792407, 1.25583031, 1.25052001, 1.24470164,
            1.2384988 , 1.2308771 , 1.22744825, 1.2275582 , 1.218964  ,
            1.2136328 , 1.2141693 , 1.21792116, 1.21125532, 1.19532303,
            1.17649556]])
    >>> points = array([[5, 4], [1, 2], [0, 0]])
    >>> mean = full(3, 1.1)
    >>> m2 = Matern(Lattice(dimension = 3,seed=7), points = points, length_scale = 4, nu = 2.5, variance = 0.01, mean=mean)
    >>> m2
    Matern (TrueMeasure Object)
        mean            [1.1 1.1 1.1]
        covariance      [[0.01  0.005 0.002]
                        [0.005 0.01  0.008]
                        [0.002 0.008 0.01 ]]
        decomp_type     PCA
    >>> from sklearn import gaussian_process as gp  #checking against scikit's Matern
    >>> kernel2 = gp.kernels.Matern(length_scale = 4, nu=2.5)
    >>> cov2 = 0.01 * kernel2.__call__(points)
    >>> isclose(cov2, m2.covariance)
    [[ True  True  True]
     [ True  True  True]
     [ True  True  True]]
    >>> distance = array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    >>> m3 = Matern(Lattice(dimension = 3,seed=7), distance = distance, length_scale = 0.5, nu = 3.5, variance = 0.01, mean=mean, decomp_type = 'Cholesky')
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

    def __init__(self, sampler, points = None, distance = None, length_scale = 1.0, nu = 1.5, variance = 1.0, mean = [], decomp_type='PCA'):
        """
        Matern kernel: calculates covariance over a metric space based only on the distance between points.
        More information can be found at [1].
        
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform. 
            points (ndarray): The positions of points on a metric space. The array
                should be of shape n x d: n points, each of dimension d.
            distance (ndarray): A 2D array where the (i, j)th value is the distance
                between the ith and jth-index points. You can pass in either a points or
                distance array; if both are passed in, distance will be used. Must
                be a symmetric matrix with diagonal values of zero.
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
        if not (isinstance(points, ndarray) or isinstance(distance, ndarray)):
            raise ParameterError("Must pass in either a points or distance ndarray.")
        if not (sampler.d == len(mean)):
            raise DimensionError("The lengths of the sampler and the mean array must be equal.")
        
        self.points = points
        self.distance = distance
        d_or_p = isinstance(distance, ndarray) #if false, distance == None, meaning we're using points
        self.d_or_p = d_or_p
        self.length_scale = length_scale
        self.nu = nu
        self.variance = variance
        
        # See [1], [2] for Matern formula
        # Replicating scikit-learn's Matern kernel using hard-coded cases, scipy.special.kv
        dimension = len(mean)
        rho = length_scale
        covariance = zeros((dimension, dimension))
        if d_or_p and (len(distance) != len(mean) or len(distance[0]) != len(mean)):
            raise DimensionError("The length and width of the distance array must equal the length of the mean array.")
        if (not d_or_p) and len(points) != len(mean):
            raise DimensionError("The number of rows of the points array must equal the length of the mean array.")
        for i in range(dimension):
            if d_or_p and distance[i][i] != 0:
                raise ParameterError("The diagonal elements of distance must equal 0.")
            for j in range(dimension):
                if d_or_p and distance[i][j] != distance [j][i]:
                    raise ParameterError("The distance array must be symmetric.")
                d = distance[i][j] if d_or_p else linalg.norm(points[i] - points[j])
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
        
