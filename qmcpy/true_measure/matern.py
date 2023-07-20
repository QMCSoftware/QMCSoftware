from .gaussian import Gaussian
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ._true_measure import TrueMeasure
from ..discrete_distribution.lattice.lattice import Lattice
from ..util import DimensionError, ParameterError
from numpy import *
#from scipy.stats import norm, multivariate_normal
from scipy.special import kv, gamma

class Matern(Gaussian):
    """
    A normal measure using a Matern kernel as the covariance matrix.
    >>> mean = full(31, 1.1)
    >>> x_values = arange(31.0) / 30.0  # [0, 1/30, ..., 1]
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
    array([[1.22818861, 1.22050757, 1.21522052, 1.2144905 , 1.21449756,
            1.22499156, 1.24161076, 1.25371086, 1.26118833, 1.2718846 ,
            1.27662595, 1.28237625, 1.28688038, 1.28466947, 1.28328019,
            1.27822864, 1.26792407, 1.25583031, 1.25052001, 1.24470164,
            1.2384988 , 1.2308771 , 1.22744825, 1.2275582 , 1.218964  ,
            1.2136328 , 1.2141693 , 1.21792116, 1.21125532, 1.19532303,
            1.17649556]])
    >>> m2 = Matern(qp.Lattice(dimension = 31,seed=7), x_values, length_scale = 0.5, nu = 2.5, variance = 0.01, mean=mean)
    >>> m2
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
    >>> m3 = Matern(qp.Lattice(dimension = 31,seed=7), x_values, length_scale = 0.5, nu = 3.5, variance = 0.01, mean=mean, decomp_type = 'Cholesky')
    >>> m3
    Matern (TrueMeasure Object)
        mean            [1.1 1.1 1.1 ... 1.1 1.1 1.1]
        covariance      [[0.01  0.01  0.01  ... 0.002 0.002 0.001]
                        [0.01  0.01  0.01  ... 0.002 0.002 0.002]
                        [0.01  0.01  0.01  ... 0.002 0.002 0.002]
                        ...
                        [0.002 0.002 0.002 ... 0.01  0.01  0.01 ]
                        [0.002 0.002 0.002 ... 0.01  0.01  0.01 ]
                        [0.001 0.002 0.002 ... 0.01  0.01  0.01 ]]
        decomp_type     CHOLESKY

     References:

        [1] scikit-learn developers, Matern kernel. 
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html.
        Accessed 2023.

        [2] Abramowitz and Stegun (1972). Handbook of Mathematical Functions with Formulas, Graphs, 
        and Mathematical Tables. ISBN 0-486-61272-4. (Accessed through Wikipedia, 
        https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function.)
    """

    def __init__(self, sampler, x_values, length_scale = 1.0, nu = 1.5, variance = 1.0, mean = [], decomp_type='PCA'):
        """
        Matern kernel: calculates covariance over a metric space based only on the distance between points.
        More information can be found at [1].
        
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
                the RBF kernel, while as nu approaches inf, it equals the absolute 
                exponential kernel. Note that nu values not in [0.5, 1.5, 2.5, inf] 
                will be ~10x slower to run.
            variance (float): The variance (or the diagonal elements of the
                covariance matrix) of the distribution at each point. Implemented
                as a constant factor of the Matern covariance. The default 
                value is 1.0.
            mean (float): mu for Normal(mu,sigma^2)
            decomp_type (str): method of decomposition either  
                "PCA" for principal component analysis or 
                "Cholesky" for cholesky decomposition.
        """
        #assert len(x_values) == len(mean)
        if not (isinstance(sampler, DiscreteDistribution) or isinstance(sampler, TrueMeasure)):
            raise ParameterError("sampler input should either be a DiscreteDistribution or TrueMeasure.")
        if not (len(x_values) == len(mean) and sampler.d == len(mean)):
            raise DimensionError("The dimensions of the position array, sampler, and means must be equal.")
        self.x_values = x_values
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
                d = abs(x_values[i] - x_values[j])
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
        return Matern(sampler, self.x_values, length_scale=self.length_scale, nu=self.nu, variance=self.variance, 
                      mean=self.mean, decomp_type=self.decomp_type)
        
