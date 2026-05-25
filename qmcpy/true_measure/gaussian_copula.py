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
from .gaussian import Gaussian
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
        >>> import numpy as np
        >>> import scipy.stats as stats
        >>> sampler = DigitalNetB2(2, seed=7)
        >>> marginals = [stats.beta(a=2, b=5), stats.gamma(a=3, scale=2)]
        >>> corr = [[1.0, 0.6], [0.6, 1.0]]
        >>> tm = GaussianCopula(sampler, marginals=marginals, correlation=corr)
        >>> np.round(tm(4), 4)
        array([[ 0.3726, 11.5225],
               [ 0.1236,  3.3233],
               [ 0.6874,  4.955 ],
               [ 0.2348,  5.3863]])
        >>> tm  # doctest: +ELLIPSIS
        GaussianCopula (AbstractTrueMeasure)
            marginals       [<...rv_continuous_frozen object at ...>
                             <...rv_continuous_frozen object at ...>]
            correlation     [[1.  0.6]
                             [0.6 1. ]]
        >>> rep_tm = GaussianCopula(DigitalNetB2(2, seed=7, replications=2), marginals=marginals, correlation=corr)
        >>> rep_tm(4).shape
        (2, 4, 2)
        >>> GaussianCopula(DigitalNetB2(1, seed=7), marginals=[stats.norm()], correlation=[[1.0]])(4).shape
        (4, 1)

    **References:**

    1.  Roger B. Nelsen. *An Introduction to Copulas*. Second Edition,
        Springer Series in Statistics, Springer, 2006.
        [doi:10.1007/0-387-28678-0](https://doi.org/10.1007/0-387-28678-0).

    2.  Mathieu Cambou, Marius Hofert, and Christiane Lemieux.
        "Quasi-random numbers for copula models."
        [arXiv:1508.03483](https://arxiv.org/abs/1508.03483).
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

        self._gaussian_transform = Gaussian(
            sampler,
            mean=np.zeros(self.d),
            covariance=self.correlation,
            decomp_type="Cholesky",
        )
        # Gaussian copula density:
        # c(u) = |R|^{-1/2} exp(-0.5 z^T (R^{-1} - I) z), z = Phi^{-1}(u).
        # The identity subtraction removes the independent standard-normal density
        # already accounted for by the marginal normal transforms.
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
        z_dep = self._gaussian_transform._transform(u)
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
