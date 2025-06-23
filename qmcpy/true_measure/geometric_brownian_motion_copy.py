from .brownian_motion import BrownianMotion
from .gaussian import Gaussian
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..discrete_distribution import DigitalNetB2
from ._true_measure import TrueMeasure
from ..util import ParameterError, _univ_repr
from numpy import exp, linspace, array, inf, minimum

class GeometricBrownianMotion(BrownianMotion):
    """
    Geometric Brownian Motion.

    >>> gbm = GeometricBrownianMotion(DigitalNetB2(4,seed=7), t_final=2, drift=0.1, diffusion=0.2)
    >>> gbm.gen_samples(2)
    array([[1.11698754, 1.74288207, 1.44791714, 0.85613458],
           [1.01036507, 0.83883012, 0.92239547, 0.91224536]])
    >>> gbm
    GeometricBrownianMotion (TrueMeasure Object)
        time_vec        [0.5 1.  1.5 2. ]
        drift           0.100
        diffusion       0.200
        mean            [0. 0. 0. 0.]
        covariance      [[0.1 0.1 0.1 0.1]
                        [0.1 0.2 0.2 0.2]
                        [0.1 0.2 0.3 0.3]
                        [0.1 0.2 0.3 0.4]]
        mean_gbm        [1.051 1.105 1.162 1.221]
        covariance_gbm  [[0.116 0.116 0.116 0.116]
                        [0.116 0.27  0.27  0.27 ]
                        [0.116 0.27  0.472 0.472]
                        [0.116 0.27  0.472 0.734]]
        decomp_type     PCA


    """

    def __init__(self, sampler, t_final=1, initial_value=1, drift=0, diffusion=1, decomp_type='PCA'):
        """
        GeometricBrownianMotion(t) = initial_value * exp[(drift - 0.5 * diffusion) * t
                                                         + \\sqrt{diffusion} * StandardBrownianMotion(t)]

        Args:
            sampler (DiscreteDistribution/TrueMeasure): A discrete distribution or true measure.
            t_final (float): End time for the geometric Brownian motion, non-negative.
            initial_value (float): Positive initial value of the process.
            drift (float): Drift coefficient.
            diffusion (float): Diffusion coefficient, positive.
            decomp_type (str): Method of decomposition, either "PCA" or "Cholesky".
        """
        
        super().__init__(sampler, t_final=t_final, drift=0, diffusion=diffusion, decomp_type=decomp_type)

        self.parameters = ['time_vec', 'drift', 'diffusion', 'mean', 'covariance', 'mean_gbm', 'covariance_gbm',
                           'decomp_type']
        self.initial_value = initial_value
        self.drift = drift
        self.diffusion = diffusion
        self.mean_gbm = self._compute_gbm_mean()
        self.covariance_gbm = self._compute_gbm_covariance()

    def _compute_gbm_mean(self):
        return self.initial_value * exp(self.drift * self.time_vec)

    def _compute_gbm_covariance(self):
        
        S0 = self.initial_value
        mu = self.drift
        t = self.time_vec
        min_t = minimum.outer(t, t)  # minimum of t_i and t_j for all pairs
        #cov_matrix = (S0 ** 2) * exp(2 * mu * min_t) * (exp(self.diffusion ** 2 * min_t) - 1)
        cov_matrix = (S0 ** 2) * exp(2 * mu * min_t) * (exp(self.diffusion * min_t) - 1)
        return cov_matrix


    def _transform(self, x):
        bm_samples = super()._transform(x)
        #return self.initial_value * exp((self.drift - 0.5 * self.diffusion ** 2) * self.time_vec + bm_samples)
        return self.initial_value * exp((self.drift - 0.5 * self.diffusion) * self.time_vec + bm_samples)

    def _spawn(self, sampler, dimension):
        return GeometricBrownianMotion(
            sampler, t_final=self.t, initial_value=self.initial_value,
            drift=self.drift, diffusion=self.diffusion, decomp_type=self.decomp_type
        )



