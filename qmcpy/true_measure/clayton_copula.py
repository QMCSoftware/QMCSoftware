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


class ClaytonCopula(AbstractTrueMeasure):
    r"""
    Clayton copula transform with user supplied marginals.

    This implementation supports general dimension for ``theta > 0``. It maps
    independent uniforms to Clayton-dependent uniforms using the conditional
    inverse / inverse Rosenblatt transform. For coordinate ``j`` after
    observing the previous ``m = j - 1`` coordinates, the conditional inverse is

    .. math::

        v = \left(1 + A
            \left(w^{-\theta/(1 + m \theta)} - 1\right)\right)^{-1/\theta},

    where ``A = 1 + sum(phi(u_i))`` over previous coordinates and
    ``phi(u) = u^{-theta} - 1``.

    then applies each marginal inverse CDF.

    Clayton copulas have positive lower-tail dependence for ``theta > 0``.

    Examples:
        >>> import numpy as np
        >>> import scipy.stats as stats
        >>> sampler = DigitalNetB2(2, seed=7)
        >>> marginals = [stats.expon(), stats.gamma(a=3)]
        >>> tm = ClaytonCopula(sampler, marginals=marginals, theta=2.0)
        >>> np.round(tm(4), 4)
        array([[1.2788, 6.1924],
               [0.1785, 1.4743],
               [4.3247, 1.9924],
               [0.5614, 2.7949]])
        >>> tm  # doctest: +ELLIPSIS
        ClaytonCopula (AbstractTrueMeasure)
            marginals       [<...rv_continuous_frozen object at ...>
                             <...rv_continuous_frozen object at ...>]
            theta           2^(1)
        >>> rep_tm = ClaytonCopula(DigitalNetB2(2, seed=7, replications=2), marginals=marginals, theta=2.0)
        >>> rep_tm(4).shape
        (2, 4, 2)
        >>> ClaytonCopula(DigitalNetB2(3, seed=7), marginals=[stats.uniform()] * 3, theta=2.0)(4).shape
        (4, 3)
        >>> ClaytonCopula(DigitalNetB2(2, seed=7), marginals=marginals, theta=1e-8)(4).shape
        (4, 2)

    **References:**

    1.  Roger B. Nelsen. *An Introduction to Copulas*. Second Edition,
        Springer Series in Statistics, Springer, 2006.
        [doi:10.1007/0-387-28678-0](https://doi.org/10.1007/0-387-28678-0).

    2.  Mathieu Cambou, Marius Hofert, and Christiane Lemieux.
        "Quasi-random numbers for copula models."
        [arXiv:1508.03483](https://arxiv.org/abs/1508.03483).

    3.  Marius Hofert, Martin Maechler, and Alexander J. McNeil.
        "Likelihood inference for Archimedean copulas in high dimensions
        under known margins." Journal of Multivariate Analysis 110,
        133-150, 2012.
        [doi:10.1016/j.jmva.2012.02.019](https://doi.org/10.1016/j.jmva.2012.02.019).
    """

    def __init__(self, sampler, marginals, theta):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                A sampler or transform whose range is the unit cube.
            marginals (list): Length d list of SciPy-like univariate
                distributions implementing ``ppf``.
            theta (float): Positive Clayton dependence parameter.
        """
        self.parameters = ["marginals", "theta"]
        self.domain = np.array([[0, 1]])
        self._parse_sampler(sampler)

        self.marginals = _validate_marginals(marginals)
        _validate_dimension(self, self.marginals)
        self.theta = self._parse_theta(theta)
        self._warned_missing_weight = False

        self.range = _build_marginal_range(self.marginals)

        super(ClaytonCopula, self).__init__()
        if len(self.marginals) != self.d:
            raise DimensionError("Length of marginals must match sampler dimension.")

    def _parse_theta(self, theta):
        try:
            theta = float(theta)
        except (TypeError, ValueError) as exc:
            raise ParameterError("theta must be a positive scalar.") from exc

        if not np.isfinite(theta) or theta <= 0:
            raise ParameterError("theta must be a positive scalar.")
        return theta

    def _log_phi(self, u):
        # Work in log space for phi(u) = u^{-theta} - 1 to avoid overflow
        # when theta is large or u is close to zero.
        u = _clip_unit_interval(u)
        a = -self.theta * np.log(u)
        return np.log(np.expm1(a))

    def _log_one_plus_sum_phi(self, log_sum_phi):
        return np.logaddexp(0.0, log_sum_phi)

    def _inverse_conditional_cdf(self, log_sum_phi, w, previous_count):
        w = _clip_unit_interval(w)
        log_a = self._log_one_plus_sum_phi(log_sum_phi)
        alpha = self.theta / (1.0 + previous_count * self.theta)
        log_delta = log_a + np.log(np.expm1(-alpha * np.log(w)))
        log_inner = np.logaddexp(0.0, log_delta)
        return _clip_unit_interval(np.exp(-log_inner / self.theta))

    def _dependent_uniforms(self, x):
        x = _clip_unit_interval(np.asarray(x, dtype=float))
        _validate_dimension(x.shape[-1], self.marginals)

        v = np.empty_like(x, dtype=float)
        v[..., 0] = x[..., 0]
        log_sum_phi = self._log_phi(v[..., 0])

        for j in range(1, self.d):
            v[..., j] = self._inverse_conditional_cdf(
                log_sum_phi,
                x[..., j],
                previous_count=j,
            )
            log_sum_phi = np.logaddexp(log_sum_phi, self._log_phi(v[..., j]))

        return _clip_unit_interval(v)

    def _transform(self, x):
        v = self._dependent_uniforms(x)
        return _apply_marginal_ppfs(v, self.marginals)

    def _unit_weight_with_warning(self, x):
        if not self._warned_missing_weight:
            warnings.warn(
                "ClaytonCopula marginals must implement 'cdf' and 'pdf' or 'logpdf' "
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

        u = _clip_unit_interval(u)
        log_u = np.log(u)
        log_sum_phi = self._log_phi(u[..., 0])
        for j in range(1, self.d):
            log_sum_phi = np.logaddexp(log_sum_phi, self._log_phi(u[..., j]))
        log_one_plus_sum_phi = self._log_one_plus_sum_phi(log_sum_phi)

        log_coefficient = np.sum(
            np.log1p(self.theta * np.arange(1, self.d, dtype=float))
        )
        log_copula_density = (
            log_coefficient
            + (-1.0 / self.theta - self.d) * log_one_plus_sum_phi
            + (-self.theta - 1.0) * np.sum(log_u, axis=-1)
        )

        return np.exp(log_copula_density + log_marginal_density)

    def _spawn(self, sampler, dimension):
        if dimension != self.d:
            raise DimensionError(
                "ClaytonCopula can only spawn with the same dimension because "
                "marginals are dimension-specific."
            )
        return ClaytonCopula(
            sampler=sampler,
            marginals=self.marginals,
            theta=self.theta,
        )
