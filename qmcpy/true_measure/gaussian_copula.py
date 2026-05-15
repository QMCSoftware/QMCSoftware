from .abstract_true_measure import AbstractTrueMeasure
from .copula import (
    _apply_marginal_ppfs,
    _build_marginal_range,
    _clip_unit_interval,
    _marginal_cdfs_and_logpdf,
    _validate_correlation_matrix,
    _validate_dimension,
    _validate_marginals,
)
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2

import numpy as np
from scipy.stats import norm
import warnings


class GaussianCopula(AbstractTrueMeasure):
    r"""
    Gaussian copula transform with user supplied univariate marginals.

    This TrueMeasure separates the dependence model from the marginal
    distributions:

    1. map independent uniforms through ``scipy.stats.norm.ppf``;
    2. inject Gaussian dependence with a Cholesky factor of the correlation;
    3. map back to dependent uniforms with ``scipy.stats.norm.cdf``;
    4. apply each marginal inverse CDF.

    The marginal objects must expose a ``ppf`` method. If they also expose
    ``cdf`` and ``pdf`` or ``logpdf``, then ``_weight`` computes the Gaussian
    copula joint density. Otherwise weights are treated as one with a warning.

    Examples:
        >>> import scipy.stats as stats
        >>> sampler = DigitalNetB2(2, seed=7)
        >>> marginals = [stats.beta(a=2, b=5), stats.gamma(a=3, scale=2)]
        >>> corr = [[1.0, 0.6], [0.6, 1.0]]
        >>> tm = GaussianCopula(sampler, marginals=marginals, correlation=corr)
        >>> tm(4).shape
        (4, 2)
    """

    def __init__(self, sampler, marginals, correlation):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                A sampler or transform whose range is the unit cube.
            marginals (list): Length d list of SciPy-like univariate
                distributions implementing ``ppf``.
            correlation (np.ndarray): d x d positive definite correlation matrix.
        """
        self.parameters = ["marginals", "correlation"]
        self.domain = np.array([[0, 1]])
        self._parse_sampler(sampler)

        self.marginals = _validate_marginals(marginals)
        _validate_dimension(self, self.marginals)
        self.correlation = _validate_correlation_matrix(correlation, self.d)

        self._chol = np.linalg.cholesky(self.correlation)
        self._corr_inv_minus_eye = np.linalg.inv(self.correlation) - np.eye(self.d)
        _, self._logdet_corr = np.linalg.slogdet(self.correlation)
        self._warned_missing_weight = False

        self.range = _build_marginal_range(self.marginals)

        super(GaussianCopula, self).__init__()
        if len(self.marginals) != self.d:
            raise DimensionError("Length of marginals must match sampler dimension.")
        if self.correlation.shape != (self.d, self.d):
            raise ValueError(
                f"correlation shape {self.correlation.shape} must match sampler dimension {self.d}."
            )

    def _transform(self, x):
        x = np.asarray(x, dtype=float)
        _validate_dimension(x.shape[-1], self.marginals)

        u = _clip_unit_interval(x)
        z = norm.ppf(u)
        z_dep = z @ self._chol.T
        u_dep = _clip_unit_interval(norm.cdf(z_dep))

        return _apply_marginal_ppfs(u_dep, self.marginals)

    def _unit_weight_with_warning(self, x):
        if not self._warned_missing_weight:
            warnings.warn(
                "GaussianCopula marginals must implement 'cdf' and 'pdf' or 'logpdf' "
                "to compute density weights. Weights will be treated as 1.",
                UserWarning,
            )
            self._warned_missing_weight = True
        return np.ones(x.shape[:-1], dtype=float)

    def _weight(self, x):
        x = np.asarray(x, dtype=float)
        try:
            u, log_marginal_density = _marginal_cdfs_and_logpdf(x, self.marginals)
        except ParameterError:
            return self._unit_weight_with_warning(x)

        z = norm.ppf(u)
        quad = np.einsum("...i,ij,...j->...", z, self._corr_inv_minus_eye, z)
        log_copula_density = -0.5 * self._logdet_corr - 0.5 * quad

        return np.exp(log_copula_density + log_marginal_density)

    def _spawn(self, sampler, dimension):
        if dimension != self.d:
            raise DimensionError(
                "GaussianCopula can only spawn with the same dimension because "
                "marginals and correlation are dimension-specific."
            )
        return GaussianCopula(
            sampler=sampler,
            marginals=self.marginals,
            correlation=self.correlation,
        )
