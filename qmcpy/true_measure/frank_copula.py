from .abstract_true_measure import AbstractTrueMeasure
from .copula import (
    _apply_marginal_ppfs,
    _build_marginal_range,
    _clip_unit_interval,
    _marginal_cdfs_and_logpdf,
    _validate_dimension,
    _validate_marginals,
)
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2

import numpy as np
import warnings


def _eulerian_coefficients(n):
    """
    Return Eulerian coefficients for Li_{-n}(z).

    For nonnegative integer n,
    Li_{-n}(z) = z * A_n(z) / (1 - z) ** (n + 1),
    where A_n is the Eulerian polynomial.
    """
    if n == 0:
        return np.array([1.0])

    coefficients = [1]
    for order in range(1, n + 1):
        next_coefficients = []
        for k in range(order):
            left = (k + 1) * coefficients[k] if k < len(coefficients) else 0
            right = (order - k) * coefficients[k - 1] if k > 0 else 0
            next_coefficients.append(left + right)
        coefficients = next_coefficients
    return np.array(coefficients, dtype=float)


class FrankCopula(AbstractTrueMeasure):
    r"""
    Frank copula transform with user supplied univariate marginals.

    This implementation supports general dimension for ``theta > 0``. Negative
    ``theta`` is supported only for the bivariate case, where the negative
    parameter Frank copula is valid. For dimensions greater than 2, ``theta``
    must be positive.

    The transform uses the inverse Rosenblatt construction for the Frank
    Archimedean copula. It maps independent uniforms to dependent uniforms by
    recursively inverting conditional CDFs, then applies each marginal inverse
    CDF.

    Examples:
        >>> import scipy.stats as stats
        >>> sampler = DigitalNetB2(3, seed=7)
        >>> marginals = [stats.norm(), stats.gamma(a=3), stats.expon()]
        >>> tm = FrankCopula(sampler, marginals=marginals, theta=5.0)
        >>> tm(4).shape
        (4, 3)
    """

    def __init__(self, sampler, marginals, theta):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                A sampler or transform whose range is the unit cube.
            marginals (list): Length d list of SciPy-like univariate
                distributions implementing ``ppf``.
            theta (float): Frank dependence parameter. Must be nonzero. Negative
                values are currently supported only for ``d=2``.
        """
        self.parameters = ["marginals", "theta"]
        self.domain = np.array([[0, 1]])
        self._parse_sampler(sampler)

        self.marginals = _validate_marginals(marginals)
        _validate_dimension(self, self.marginals)
        self.theta = self._parse_theta(theta)

        self._expm1_neg_theta = np.expm1(-self.theta)
        if self._expm1_neg_theta == 0 or not np.isfinite(self._expm1_neg_theta):
            raise ParameterError("theta is too close to 0 or too large in magnitude.")
        self._alpha = -self._expm1_neg_theta
        self._eulerian_cache = {}
        self._warned_missing_weight = False

        self.range = _build_marginal_range(self.marginals)

        super(FrankCopula, self).__init__()
        assert len(self.marginals) == self.d

    def _parse_theta(self, theta):
        try:
            theta = float(theta)
        except (TypeError, ValueError) as exc:
            raise ParameterError("theta must be a finite nonzero scalar.") from exc

        if not np.isfinite(theta) or theta == 0:
            raise ParameterError("theta must be a finite nonzero scalar.")
        if theta < 0 and self.d != 2:
            raise ParameterError("theta < 0 is only supported for d=2 FrankCopula.")
        return theta

    def _eulerian_coefficients(self, n):
        if n not in self._eulerian_cache:
            self._eulerian_cache[n] = _eulerian_coefficients(n)
        return self._eulerian_cache[n]

    def _q(self, u):
        u = _clip_unit_interval(u)
        q = np.expm1(-self.theta * u) / self._expm1_neg_theta
        eps = np.finfo(float).eps
        return np.clip(q, eps, 1.0 - eps)

    def _z_from_q_product(self, q_product):
        z = self._alpha * q_product
        eps = np.finfo(float).eps
        tiny = np.finfo(float).tiny

        if self.theta > 0:
            return np.clip(z, tiny, 1.0 - eps)
        return np.minimum(z, -tiny)

    def _log_abs_polylog_negative_order(self, z, n):
        z = np.asarray(z, dtype=float)
        coefficients = self._eulerian_coefficients(n)

        polynomial = np.zeros_like(z, dtype=float)
        for coefficient in coefficients[::-1]:
            polynomial = polynomial * z + coefficient

        tiny = np.finfo(float).tiny
        return (
            np.log(np.maximum(np.abs(z), tiny))
            + np.log(np.maximum(np.abs(polynomial), tiny))
            - (n + 1.0) * np.log1p(-z)
        )

    def _conditional_cdf(self, q_previous, v, derivative_order):
        q_v = self._q(v)
        z_previous = self._z_from_q_product(q_previous)
        z_new = self._z_from_q_product(q_previous * q_v)
        n = derivative_order - 1

        log_conditional = (
            self._log_abs_polylog_negative_order(z_new, n)
            - self._log_abs_polylog_negative_order(z_previous, n)
        )
        return np.clip(np.exp(log_conditional), 0.0, 1.0)

    def _inverse_conditional_cdf(self, q_previous, w, derivative_order):
        w = _clip_unit_interval(w)
        eps = np.finfo(float).eps
        lo = np.full_like(w, eps, dtype=float)
        hi = np.full_like(w, 1.0 - eps, dtype=float)

        for _ in range(60):
            mid = (lo + hi) / 2.0
            conditional_mid = self._conditional_cdf(q_previous, mid, derivative_order)
            lo = np.where(conditional_mid < w, mid, lo)
            hi = np.where(conditional_mid >= w, mid, hi)

        return _clip_unit_interval((lo + hi) / 2.0)

    def _dependent_uniforms(self, x):
        x = _clip_unit_interval(np.asarray(x, dtype=float))
        _validate_dimension(x.shape[-1], self.marginals)

        v = np.empty_like(x, dtype=float)
        v[..., 0] = x[..., 0]
        q_product = self._q(v[..., 0])

        for j in range(1, self.d):
            v[..., j] = self._inverse_conditional_cdf(
                q_product,
                x[..., j],
                derivative_order=j,
            )
            q_product = q_product * self._q(v[..., j])

        return _clip_unit_interval(v)

    def _transform(self, x):
        v = self._dependent_uniforms(x)
        return _apply_marginal_ppfs(v, self.marginals)

    def _unit_weight_with_warning(self, x):
        if not self._warned_missing_weight:
            warnings.warn(
                "FrankCopula marginals must implement 'cdf' and 'pdf' or 'logpdf' "
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

        q_product = np.prod(self._q(u), axis=-1)
        z = self._z_from_q_product(q_product)

        log_abs_psi_derivative = (
            self._log_abs_polylog_negative_order(z, self.d - 1)
            - np.log(abs(self.theta))
        )
        log_abs_phi_prime = (
            np.log(abs(self.theta))
            - self.theta * u
            - np.log(np.abs(np.expm1(-self.theta * u)))
        )
        log_copula_density = log_abs_psi_derivative + np.sum(log_abs_phi_prime, axis=-1)

        return np.exp(log_copula_density + log_marginal_density)

    def _spawn(self, sampler, dimension):
        if dimension != self.d:
            raise DimensionError(
                "FrankCopula can only spawn with the same dimension because "
                "marginals are dimension-specific."
            )
        return FrankCopula(
            sampler=sampler,
            marginals=self.marginals,
            theta=self.theta,
        )
