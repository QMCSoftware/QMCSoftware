from .brownian_motion import BrownianMotion
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError
from numpy import exp, zeros, minimum, array, sqrt


class GeometricBrownianMotion(BrownianMotion):
    """
    Geometric Brownian Motion.

    >>> gbm = GeometricBrownianMotion(DigitalNetB2(4,seed=7), t_final=2, drift=0.1, diffusion=0.2)
    >>> gbm.gen_samples(2)
    array([[1.09360288, 1.38880698, 1.33046088, 1.0947602 ],
           [1.0456216 , 1.0014045 , 1.08749136, 1.12628555]])
    >>> gbm
    GeometricBrownianMotion (TrueMeasure Object)
        time_vec        [0.5 1.  1.5 2. ]
        drift           0.100
        diffusion       0.200
        mean_gbm        [1.051 1.105 1.162 1.221]
        covariance_gbm  [[0.022 0.023 0.025 0.026]
                        [0.023 0.05  0.052 0.055]
                        [0.025 0.052 0.083 0.088]
                        [0.026 0.055 0.088 0.124]]
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

        self.parameters = ['time_vec', 'drift', 'diffusion', 'mean_gbm', 'covariance_gbm', 'decomp_type']
        self.initial_value = initial_value
        self.drift = drift
        self.diffusion = diffusion
        self.mean_gbm = self._compute_gbm_mean()
        self.covariance_gbm = self._compute_gbm_covariance()
        self._validate_input()

    def _compute_gbm_mean(self):
        return self.initial_value * exp(self.drift * self.time_vec)

    def _compute_gbm_covariance(self):
        S0 = self.initial_value
        mu = self.drift
        t = array(self.time_vec) 
        # Vectorization using broadcasting
        t_sum = t[:, None] + t[None, :]  # t[i] + t[j] for all i,j
        t_min = minimum.outer(t, t)      # min(t[i], t[j]) for all i,j
        cov_matrix = (S0 ** 2) * exp(mu * t_sum) * (exp(self.diffusion**2 * t_min) - 1)
        
        return cov_matrix
        

    def _transform(self, x):
        bm_samples = super()._transform(x)
        # Note: bm_samples already includes the diffusion scaling from the parent BrownianMotion class
        # So we don't multiply by self.diffusion again here
        exponent = (self.drift - 0.5 * self.diffusion**2) * self.time_vec + bm_samples
        samples = self.initial_value * exp(exponent)
        self._validate_samples(samples, strict=True)
        return samples
  

    def _spawn(self, sampler, dimension):
        return GeometricBrownianMotion(
            sampler, t_final=self.t, initial_value=self.initial_value,
            drift=self.drift, diffusion=self.diffusion, decomp_type=self.decomp_type
        )

    def _validate_input(self):
        """
        Validates the input parameters of the GeometricBrownianMotion class.

        Raises:
            ValueError: If the end time `t_final' is negative.
            ValueError: If the diffusion coefficient is less than or equal to zero.
            ValueError: If the initial value is less than or equal to zero.
            ParameterError: If the decomposition type is not 'PCA' or 'Cholesky'.
        """
        if self.t < 0:
            raise ValueError(f"End time 't_final' must be non-negative. It should not be {self.t}.")
        if self.diffusion <= 0:
            raise ValueError(f"Diffusion coefficient must be positive. It should not be {self.diffusion}.")
        if self.initial_value <= 0:
            raise ValueError(f"Initial value must be positive. It should not be {self.initial_value}.")
        if self.decomp_type.upper() not in ['PCA', 'CHOLESKY']:
            raise ParameterError(f"Decomposition type must be 'PCA' or 'Cholesky'. It should not be {self.decomp_type}.")

    def _validate_samples(self, samples, strict=True):
        """
        Validate that generated GBM samples meet mathematical requirements.
        """
        min_val = samples.min()
        max_val = samples.max()
        num_negative = (samples <= 0).sum()
        total_samples = samples.size
        
        validation_results = {
            'is_valid': num_negative == 0,
            'min_value': min_val,
            'max_value': max_val,
            'num_negative': num_negative,
            'total_samples': total_samples,
            'fraction_negative': num_negative / total_samples if total_samples > 0 else 0
        }
        
        if num_negative > 0:
            error_msg = (f"GeometricBrownianMotion validation FAILED: "
                        f"{num_negative}/{total_samples} ({100*validation_results['fraction_negative']:.2f}%) "
                        f"samples are non-positive (min: {min_val:.6f}). "
                        f"This violates the mathematical definition of GBM where S(t) = S0 * exp(...) > 0.")
            
            if strict:
                raise ValueError(error_msg)
            else:
                import warnings
                warnings.warn(error_msg)
        
        return validation_results



