from .brownian_motion import BrownianMotion
# from .gaussian import Gaussian
from numpy import exp, linspace, array, inf, minimum
# from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..discrete_distribution import DigitalNetB2
# from ._true_measure import TrueMeasure
# from ..util import ParameterError, _univ_repr
# from numpy import *

class GeometricBrownianMotion(BrownianMotion):
    """
    Geometric Brownian Motion inheriting from BrownianMotion.
    """

    def __init__(self, sampler, t_final=1, initial_value=1, drift=0, diffusion=1, decomp_type='PCA'):
        """
        GeometricBrownianMotion(t) = initial_value * exp[(drift - 0.5 * diffusion) * t
                                                         + \sqrt{diffusion} * StandardBrownianMotion(t)]

        Args:
            sampler (DiscreteDistribution/TrueMeasure): A discrete distribution or true measure.
            t_final (float): End time for the geometric Brownian motion, non-negative.
            initial_value (float): Positive initial value of the process.
            drift (float): Drift coefficient.
            diffusion (float): Diffusion coefficient, positive.
            decomp_type (str): Method of decomposition, either "PCA" or "Cholesky".
        """
        
        super().__init__(sampler, t_final=t_final, drift=0, diffusion=diffusion, decomp_type=decomp_type)

        
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
        cov_matrix = (S0 ** 2) * exp(2 * mu * min_t) * (exp(self.diffusion ** 2 * min_t) - 1)
        return cov_matrix


    def _transform(self, x):
        bm_samples = super()._transform(x)
        return self.initial_value * exp((self.drift - 0.5 * self.diffusion ** 2) * self.time_vec + bm_samples)

    def _spawn(self, sampler, dimension):
        return GeometricBrownianMotion(
            sampler, t_final=self.t, initial_value=self.initial_value,
            drift=self.drift, diffusion=self.diffusion, decomp_type=self.decomp_type
        )



