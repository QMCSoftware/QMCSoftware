from .brownian_motion import BrownianMotion
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError
from numpy import exp, zeros, minimum, array, sqrt, log, pi, linalg, eye
from scipy.stats import multivariate_normal


class GeometricBrownianMotion(BrownianMotion):
    r"""
    A Geometric Brownian Motion (GBM) with initial value $S_0$, drift $\gamma$, and diffusion $\sigma^2$ is 

    $$\mathrm{GBM}(t) = S_0 \exp[(\gamma - \sigma/2) t + \sigma \mathrm{BM}(t)]$$

    where BM is a Brownian Motion drift $\gamma$ and diffusion $\sigma^2$. 
    
    Examples:
        >>> gbm = GeometricBrownianMotion(DigitalNetB2(4,seed=7), t_final=2, drift=0.1, diffusion=0.2)
        >>> gbm.gen_samples(2)
        array([[0.92343761, 1.42069027, 1.30851806, 0.99133819],
               [0.7185916 , 0.42028013, 0.42080335, 0.4696196 ]])
        >>> gbm
        GeometricBrownianMotion (AbstractTrueMeasure)
            time_vec        [0.5 1.  1.5 2. ]
            drift           0.100
            diffusion       0.200
            mean_gbm        [1.051 1.105 1.162 1.221]
            covariance_gbm  [[0.116 0.122 0.128 0.135]
                             [0.122 0.27  0.284 0.299]
                             [0.128 0.284 0.472 0.496]
                             [0.135 0.299 0.496 0.734]]
            decomp_type     PCA
    """

    def __init__(self, sampler, t_final=1, initial_value=1, drift=0, diffusion=1, decomp_type='PCA'):
        r"""
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A discrete distribution or true measure.
            t_final (float): End time for the geometric Brownian motion, non-negative.
            initial_value (float): Positive initial value of the process, $S_0$.
            drift (float): Drift coefficient $\gamma$.
            diffusion (float): Positive diffusion coefficient $\sigma^2$.
            decomp_type (str): Method of decomposition, either "PCA" or "Cholesky".
        """
        super().__init__(sampler, t_final=t_final, drift=0, diffusion=diffusion, decomp_type=decomp_type)
        self.parameters = ['time_vec', 'drift', 'diffusion', 'mean_gbm', 'covariance_gbm', 'decomp_type']
        self.initial_value = initial_value
        self.drift = drift
        self.diffusion = diffusion
        self.mean_gbm = self._compute_gbm_mean()
        self.covariance_gbm = self._compute_gbm_covariance()
        self._setup_lognormal_distribution()
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
        cov_matrix = (S0 ** 2) * exp(mu * t_sum) * (exp(self.diffusion * t_min) - 1)
        
        return cov_matrix
        

    def _transform(self, x):
        bm_samples = super()._transform(x)
        # Note: bm_samples already includes the diffusion scaling from the parent BrownianMotion class
        # So we don't multiply by self.diffusion again here
        exponent = (self.drift - 0.5 * self.diffusion) * self.time_vec + bm_samples
        samples = self.initial_value * exp(exponent)
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

    def _validate_samples(self, samples, strict=False):
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

    def _setup_lognormal_distribution(self):
        """Setup scipy multivariate normal for the log-transformed variables."""
        # Mean of log(S(t)/S0): (drift - 0.5*diffusion) * t
        log_mean = (self.drift - 0.5 * self.diffusion) * self.time_vec
        
        # Covariance of log returns: diffusion * min(t_i, t_j)
        time_matrix = minimum.outer(self.time_vec, self.time_vec)
        log_cov = self.diffusion * time_matrix
        
        self.log_mvn_scipy = multivariate_normal(mean=log_mean, cov=log_cov, allow_singular=True)

    def _weight(self, x):
        """
        Compute PDF of multivariate log-normal distribution.
        For log-normal: f(x) = (1/∏x_i) * φ(log(x/S0)) where φ is multivariate normal PDF.
        
        Args:
            x (ndarray): GBM sample paths of shape (n_samples, n_timepoints)
            
        Returns:
            ndarray: PDF values for each sample path
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Transform to log space: log(x/S0)
        log_transformed = log(x / self.initial_value)
        
        # Get PDF of underlying multivariate normal
        normal_pdf = self.log_mvn_scipy.pdf(log_transformed)
        
        # Apply Jacobian transformation: divide by product of x values
        jacobian = 1.0 / x.prod(axis=1)
        
        return normal_pdf * jacobian
