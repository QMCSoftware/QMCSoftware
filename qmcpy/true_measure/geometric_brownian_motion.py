from .gaussian import Gaussian
from ..discrete_distribution import DigitalNetB2
from numpy import exp, linspace, array, inf


class GeometricBrownianMotion(Gaussian):
    """
    Geometric Brownian motion.

    >>> gbm = GeometricBrownianMotion(DigitalNetB2(4, seed=7), t_final=2, drift=0.1, diffusion=0.2)
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
        decomp_type     PCA
    """

    def __init__(self, sampler, t_final=1, initial_value=1, drift=0, diffusion=1, decomp_type='PCA'):
        """
        GeometricBrownianMotion(t) = (initial_value) * exp((drift - 0.5 * diffusion^2) * t + diffusion * StandardBrownianMotion(t))

        Args:
            sampler (DiscreteDistribution/TrueMeasure): A
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
            t_final (float): end time for the geometric Brownian motion.
            initial_value (float): initial value of the process
            drift (float): drift coefficient
            diffusion (float): diffusion coefficient (also known as volatility)
            decomp_type (str): method of decomposition either
                "PCA" for principal component analysis or
                "Cholesky" for cholesky decomposition.
        """
        self.parameters = ['time_vec', 'drift', 'diffusion', 'mean', 'covariance', 'decomp_type']
        self.domain = array([[0, 1]])
        self._parse_sampler(sampler)
        self.t = t_final
        self.initial_value = initial_value
        self.drift = drift
        self.diffusion = diffusion
        self.time_vec = linspace(self.t / self.d, self.t, self.d)

        # Validate t_final >= 0, diffusion > 0, and initial_value > 0
        if self.t < 0:
            raise ValueError("End time must be non-negative")
        if self.diffusion <= 0:
            raise ValueError("Diffusion coefficient must be positive")
        if self.initial_value <= 0:
            raise ValueError("Initial value must be positive")

        # Compute mean and covariance of standard Brownian motion, log(S(t) / initial_value))
        mean_bm = (self.drift - 0.5 * self.diffusion ** 2) * self.time_vec
        covariance_bm = self.diffusion**2 * array([[min(self.time_vec[i], self.time_vec[j]) for i in range(self.d)]
                                                   for j in range(self.d)])
        self._parse_gaussian_params(mean_bm, covariance_bm, decomp_type)
        self.range = array([[-inf, inf]])

    def _transform(self, x):
        # generate standard Brownian motion samples
        normal_samples = super()._transform(x)
        # Transforms above samples into geometric Brownian motion samples
        return self.initial_value * exp(normal_samples)

    def _spawn(self, sampler, dimension):
        return GeometricBrownianMotion(sampler, t_final=self.t, drift=self.drift, diffusion=self.diffusion,
                                       decomp_type=self.decomp_type)