from .gaussian import Gaussian
from ..discrete_distribution.abstract_discrete_distribution import AbstractDiscreteDistribution
from .abstract_true_measure import AbstractTrueMeasure
from ..discrete_distribution.lattice.lattice import Lattice
from ..util import DimensionError, ParameterError
import numpy as np
from scipy.special import kv, gamma

class Matern(Gaussian):
    r"""
    A `Gaussian` with Matérn covariance kernel, see  
        [https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html) and  
        [https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)

    Examples:
        >>> x_values = np.array([0, 1, 2])
        >>> mean = np.full(3, 1.1)
        >>> m3 = Matern(Lattice(dimension = 3,seed=7), x_values, length_scale = 0.5, nu = 3.5, variance = 0.01, mean=mean, decomp_type = 'Cholesky')
        >>> m3
        Matern (AbstractTrueMeasure)
            mean            [1.1 1.1 1.1]
            covariance      [[1.000e-02 1.378e-03 3.432e-05]
                            [1.378e-03 1.000e-02 1.378e-03]
                            [3.432e-05 1.378e-03 1.000e-02]]
            decomp_type     CHOLESKY
        
        >>> mean = np.full(31, 1.1)
        >>> x_values = np.arange(31.0) / 30.0 #[0, 1/30, ..., 1]
        >>> m = Matern(Lattice(dimension=31, seed=7), x_values, length_scale = 0.5, variance = 0.01, mean=mean)
        >>> m
        Matern (AbstractTrueMeasure)
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
        array([[0.98982459, 0.99436558, 0.9985705 , 0.99537541, 0.98332323,
                0.97012349, 0.96397891, 0.9563429 , 0.95509076, 0.95367365,
                0.94488316, 0.93887217, 0.9447937 , 0.94984784, 0.95086118,
                0.95391034, 0.95711649, 0.96437995, 0.96164923, 0.95571374,
                0.95318471, 0.95162953, 0.9486336 , 0.94254554, 0.93762367,
                0.93127782, 0.93008568, 0.93477752, 0.9510726 , 0.96926194,
                0.98501258]])
        
        With independent replications 

        >>> x =  Matern(Lattice(dimension=3,seed=7,replications=2),points=np.array([0, 1, 2]),length_scale=0.5,nu=3.5,variance=0.01,mean=np.full(3,1.1),decomp_type='Cholesky')(4)
        >>> x.shape 
        (2, 4, 3)
        >>> x
        array([[[0.9292457 , 1.09831694, 1.06939215],
                [1.11101646, 0.96703953, 1.19238447],
                [1.04578587, 1.05093947, 0.97722352],
                [1.181989  , 1.2086904 , 1.14395342]],
        <BLANKLINE>
               [[1.1391077 , 1.15639037, 0.98369902],
                [0.99726585, 1.00131231, 1.11446066],
                [1.22937832, 1.10455322, 1.20383499],
                [1.07521619, 1.25641254, 1.08573892]]])
    """

    def __init__(self, sampler, points, length_scale = 1.0, nu = 1.5, variance = 1.0, mean = [], decomp_type='PCA'):
        """
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): A 
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
        if not (isinstance(sampler, AbstractDiscreteDistribution) or isinstance(sampler, AbstractTrueMeasure)):
            raise ParameterError("sampler input should either be an AbstractDiscreteDistribution or AbstractTrueMeasure.")
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
        
