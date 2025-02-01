from .gaussian import Gaussian
from ..discrete_distribution import DigitalNetB2
from numpy import exp, linspace, array, inf, minimum


class GeometricBrownianMotion(Gaussian):
    """
    Geometric Brownian motion.

    >>> gbm = GeometricBrownianMotion(DigitalNetB2(4,seed=7), t_final=2, drift=0.1, diffusion=0.2)
    >>> gbm.gen_samples(2)
    array([[0.93056515, 1.00539546, 1.06060187, 1.22569294],
           [1.26503416, 1.14339958, 1.11138338, 1.36923737]])
    >>> gbm
    GeometricBrownianMotion (TrueMeasure Object)
        time_vec        [0.5 1.  1.5 2. ]
        drift           0.100
        diffusion       0.200
        mean            [0.04 0.08 0.12 0.16]
        covariance      [[0.02 0.02 0.02 0.02]
                        [0.02 0.04 0.04 0.04]
                        [0.02 0.04 0.06 0.06]
                        [0.02 0.04 0.06 0.08]]
        mean_gbm        [1.051 1.105 1.162 1.221]
        covariance_gbm  [[0.022 0.022 0.022 0.022]
                        [0.022 0.05  0.05  0.05 ]
                        [0.022 0.05  0.083 0.083]
                        [0.022 0.05  0.083 0.124]]
        decomp_type     PCA
    """

    def __init__(self, sampler, t_final=1, initial_value=1, drift=0, diffusion=1, decomp_type='PCA'):
        """
        GeometricBrownianMotion(t) = initial_value * exp[(drift - 0.5 * diffusion^2) * t
                                                         + diffusion * StandardBrownianMotion(t)]

        Args:
            sampler (DiscreteDistribution/TrueMeasure): A
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
            t_final (float): end time for the geometric Brownian motion, non-negative
            initial_value (float): positive initial value of the process
            drift (float): drift coefficient
            diffusion (float): diffusion coefficient (also known as volatility), positive
            decomp_type (str): method of decomposition either
                "PCA" for principal component analysis or
                "Cholesky" for Cholesky decomposition.
        """
        self.parameters = ['time_vec', 'drift', 'diffusion', 'mean', 'covariance', 'mean_gbm', 'covariance_gbm',
                           'decomp_type']
        self.domain = array([[0, 1]])
        self._parse_sampler(sampler)
        self.t = t_final
        self.initial_value = initial_value
        self.drift = drift
        self.diffusion = diffusion
        self.time_vec = linspace(self.t / self.d, self.t, self.d)
        self.decomp_type = decomp_type

        # Validate input
        self._validate_input()

        # Compute mean and covariance of standard Brownian motion, log(S(t) / initial_value))
        mean_bm = (self.drift - 0.5 * self.diffusion ** 2) * self.time_vec
        
        # covariance_bm = self.diffusion**2 * array([[min(self.time_vec[i], self.time_vec[j]) for i in range(self.d)]
        #                                              for j in range(self.d)]) 
        covariance_bm = self.diffusion * array([[min(self.time_vec[i], self.time_vec[j]) for i in range(self.d)]
                                                     for j in range(self.d)])
        self._parse_gaussian_params(mean_bm, covariance_bm, decomp_type)
        self.range = array([[-inf, inf]])

        # Compute the mean or expected value of the Brownian motion process.
        self.mean_gbm = self._compute_gbm_mean()
        # Compute covariance matrix of the geometric Brownian motion process.
        self.covariance_gbm = self._compute_gbm_covariance()

    def _compute_gbm_covariance(self):
        """
        Covariance between S(t_i) and S(t_j) for any t_i and t_j in time_vec with evenly spaced time points.

        Reference: https://en.wikipedia.org/wiki/Geometric_Brownian_motion#Multivariate_version
        """
        S0 = self.initial_value
        mu = self.drift
        sigma = self.diffusion
        t = self.time_vec
        min_t = minimum.outer(t, t)  # minimum of t_i and t_j for all pairs
        cov_matrix = (S0 ** 2) * exp(2 * mu * min_t) * (exp(sigma ** 2 * min_t) - 1)
        return cov_matrix

    def _compute_gbm_mean(self):
        """
        Expectation of GeometricBrownianMotion(t) for t in time_vec

        Reference: https://en.wikipedia.org/wiki/Geometric_Brownian_motion#Properties_of_GBM
        """
        return self.initial_value * exp(self.drift * self.time_vec)

    def _validate_input(self):
        """
        Validates the input parameters of the GeometricBrownianMotion class.

        Raises:
            ValueError: If the end time `t_final' is negative.
            ValueError: If the diffusion coefficient is less than or equal to zero.
            ValueError: If the initial value is less than or equal to zero.
            ValueError: If the decomposition type is not 'PCA' or 'Cholesky'.
        """
        if self.t < 0:
            raise ValueError(f"End time 't_final' must be non-negative. It should not be {self.t}.")
        if self.diffusion <= 0:
            raise ValueError(f"Diffusion coefficient must be positive. It should not be {self.diffusion}.")
        if self.initial_value <= 0:
            raise ValueError(f"Initial value must be positive. It should not be {self.initial_value}.")
        if self.decomp_type.upper() not in ['PCA', 'CHOLESKY']:
            raise ValueError(f"Decomposition type must be 'PCA' or 'Cholesky'. It should not be {self.decomp_type}.")

    def _transform(self, x):
        # generate standard Brownian motion samples
        normal_samples = super()._transform(x)
        # Transforms above samples into geometric Brownian motion samples
        return self.initial_value * exp(normal_samples)

    def _spawn(self, sampler, dimension):
        return GeometricBrownianMotion(sampler, t_final=self.t, drift=self.drift, diffusion=self.diffusion,
                                       decomp_type=self.decomp_type)