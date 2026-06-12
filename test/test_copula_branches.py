import warnings

import numpy as np
import pytest
import scipy.stats as stats

from qmcpy.discrete_distribution import DigitalNetB2
from qmcpy.true_measure import (
    ClaytonCopula,
    Copula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
)
from qmcpy.true_measure.copula import (
    _apply_marginal_ppfs,
    _build_marginal_range,
    _clip_unit_interval,
    _marginal_cdfs_and_logpdf,
    _validate_correlation_matrix,
    _validate_dimension,
    _validate_marginals,
)
from qmcpy.util import DimensionError, MethodImplementationError, ParameterError


class PPFOnlyMarginal:
    def ppf(self, u):
        return np.asarray(u, dtype=float)


class NonCallablePPFMarginal:
    ppf = 1.0


class UnitPDFMarginal:
    def ppf(self, u):
        return np.asarray(u, dtype=float)

    def cdf(self, x):
        return np.asarray(x, dtype=float)

    def pdf(self, x):
        return np.ones_like(np.asarray(x, dtype=float))


class CDFOnlyMarginal(PPFOnlyMarginal):
    def cdf(self, x):
        return np.asarray(x, dtype=float)


class BadIntervalMarginal(PPFOnlyMarginal):
    def interval(self, confidence):
        raise ValueError("interval unavailable")


class BadRangeMarginal:
    def ppf(self, u):
        raise ValueError("ppf unavailable")


def _identity_correlation(d):
    return np.eye(d)


def _make_copula(copula_cls, dimension=2, marginals=None):
    if marginals is None:
        marginals = [stats.uniform()] * dimension

    sampler = DigitalNetB2(dimension, seed=113)
    if copula_cls is GaussianCopula:
        return copula_cls(
            sampler,
            marginals=marginals,
            correlation=_identity_correlation(dimension),
        )
    if copula_cls is StudentTCopula:
        return copula_cls(
            sampler,
            marginals=marginals,
            correlation=_identity_correlation(dimension),
            df=4,
        )
    if copula_cls is ClaytonCopula:
        return copula_cls(sampler, marginals=marginals, theta=1.5)
    if copula_cls is GumbelCopula:
        return copula_cls(sampler, marginals=marginals, theta=1.5)
    if copula_cls is FrankCopula:
        return copula_cls(sampler, marginals=marginals, theta=4.0)
    raise TypeError("unsupported copula class")


def test_base_copula_rejects_unimplemented_transform():
    tm = Copula(DigitalNetB2(2, seed=101), marginals=[stats.uniform(), stats.uniform()])

    with pytest.raises(MethodImplementationError):
        tm.copula_transform(np.full((3, 2), 0.5))


def test_base_copula_rejects_invalid_sampler():
    with pytest.raises(ParameterError, match="sampler"):
        Copula(object(), marginals=[stats.uniform()])


def test_validate_marginals_error_branches():
    with pytest.raises(ParameterError, match="marginals"):
        _validate_marginals(None)

    with pytest.raises(ParameterError, match="at least one"):
        _validate_marginals([])

    with pytest.raises(ParameterError, match="ppf"):
        _validate_marginals([NonCallablePPFMarginal()])


def test_validate_dimension_error_branches():
    with pytest.raises(DimensionError, match="integer dimension"):
        _validate_dimension(object(), [stats.uniform()])

    with pytest.raises(DimensionError, match="marginals"):
        _validate_dimension(3, [stats.uniform(), stats.uniform()])


def test_apply_marginal_ppfs_clips_endpoints_and_checks_dimension():
    transformed = _apply_marginal_ppfs(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        [stats.norm(), stats.norm()],
    )

    assert transformed.shape == (2, 2)
    assert np.all(np.isfinite(transformed))

    with pytest.raises(DimensionError, match="marginals"):
        _apply_marginal_ppfs(np.full((2, 3), 0.5), [stats.uniform(), stats.uniform()])


def test_marginal_range_falls_back_when_interval_or_ppf_fails():
    ranges = _build_marginal_range([BadIntervalMarginal(), BadRangeMarginal()])

    assert ranges.shape == (2, 2)
    assert np.all(np.isfinite(ranges[0]))
    np.testing.assert_allclose(ranges[1], [-np.inf, np.inf])


def test_marginal_cdfs_and_logpdf_pdf_branch_and_errors():
    x = np.array([[0.25, 0.75], [0.4, 0.6]])
    u, log_density = _marginal_cdfs_and_logpdf(
        x,
        [UnitPDFMarginal(), UnitPDFMarginal()],
    )

    np.testing.assert_allclose(u, x)
    np.testing.assert_allclose(log_density, np.zeros(2))

    with pytest.raises(ParameterError, match="cdf"):
        _marginal_cdfs_and_logpdf(x, [PPFOnlyMarginal(), UnitPDFMarginal()])

    with pytest.raises(ParameterError, match="pdf"):
        _marginal_cdfs_and_logpdf(x, [CDFOnlyMarginal(), UnitPDFMarginal()])


def test_validate_correlation_matrix_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        _validate_correlation_matrix([[1.0, np.nan], [np.nan, 1.0]], 2)


def test_clip_unit_interval_uses_machine_epsilon():
    clipped = _clip_unit_interval(np.array([0.0, 0.5, 1.0]))
    eps = np.finfo(float).eps

    np.testing.assert_allclose(clipped, [eps, 0.5, 1.0 - eps])


@pytest.mark.parametrize(
    "copula_cls",
    [GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula],
)
def test_copula_transform_outputs_dependent_uniforms_in_unit_cube(copula_cls):
    tm = _make_copula(copula_cls, dimension=3)
    u = np.array(
        [
            [0.1, 0.3, 0.7],
            [0.5, 0.5, 0.5],
            [0.9, 0.8, 0.2],
        ]
    )

    v = tm.copula_transform(u)

    assert v.shape == u.shape
    assert np.all(np.isfinite(v))
    assert np.all((0.0 <= v) & (v <= 1.0))


@pytest.mark.parametrize(
    "copula_cls",
    [GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula],
)
def test_copula_weight_fallback_warns_once_when_density_methods_are_missing(
    copula_cls,
):
    tm = _make_copula(
        copula_cls,
        dimension=2,
        marginals=[PPFOnlyMarginal(), PPFOnlyMarginal()],
    )
    x = np.full((4, 2), 0.5)

    with pytest.warns(UserWarning, match="Weights will be treated as 1"):
        weights = tm._weight(x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        second_weights = tm._weight(x)

    np.testing.assert_allclose(weights, np.ones(4))
    np.testing.assert_allclose(second_weights, np.ones(4))
    assert caught == []


def test_student_t_weight_falls_back_when_multivariate_t_is_unavailable():
    tm = StudentTCopula(
        DigitalNetB2(2, seed=115),
        marginals=[stats.norm(), stats.norm()],
        correlation=np.eye(2),
        df=4,
    )
    tm._mvt_scipy = None

    with pytest.warns(UserWarning, match="Weights will be treated as 1"):
        weights = tm._weight(np.full((3, 2), 0.25))

    np.testing.assert_allclose(weights, np.ones(3))


def test_gaussian_weight_uses_pdf_branch_when_logpdf_is_unavailable():
    tm = GaussianCopula(
        DigitalNetB2(2, seed=117),
        marginals=[UnitPDFMarginal(), UnitPDFMarginal()],
        correlation=[[1.0, 0.4], [0.4, 1.0]],
    )

    weights = tm._weight(np.array([[0.25, 0.5], [0.75, 0.5]]))

    assert weights.shape == (2,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)


def test_gumbel_theta_one_weight_is_independent_marginal_density():
    tm = GumbelCopula(
        DigitalNetB2(2, seed=119),
        marginals=[stats.gamma(a=2.0), stats.expon()],
        theta=1.0,
    )
    x = np.array([[1.0, 0.5], [2.0, 1.5]])
    expected = stats.gamma(a=2.0).pdf(x[:, 0]) * stats.expon().pdf(x[:, 1])

    weights = tm._weight(x)

    np.testing.assert_allclose(weights, expected)


def test_gen_copula_samples_composed_transform_branch():
    inner = GaussianCopula(
        DigitalNetB2(2, seed=121),
        marginals=[stats.uniform(), stats.uniform()],
        correlation=[[1.0, 0.3], [0.3, 1.0]],
    )
    outer = ClaytonCopula(inner, marginals=[stats.uniform(), stats.uniform()], theta=1.5)

    v = outer.gen_copula_samples(n_min=4, n_max=8)

    assert v.shape == (4, 2)
    assert np.all(np.isfinite(v))
    assert np.all((0.0 <= v) & (v <= 1.0))


@pytest.mark.parametrize(
    "copula_cls",
    [GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula],
)
def test_copula_spawn_same_dimension_and_reject_different_dimension(copula_cls):
    tm = _make_copula(copula_cls, dimension=2)

    spawned = tm.spawn(s=1, dimensions=[2])
    assert len(spawned) == 1
    assert isinstance(spawned[0], copula_cls)
    assert spawned[0](4).shape == (4, 2)

    with pytest.raises(DimensionError):
        tm._spawn(DigitalNetB2(3, seed=123), 3)


def test_frank_one_dimensional_weight_covers_zero_order_eulerian_term():
    tm = FrankCopula(
        DigitalNetB2(1, seed=125),
        marginals=[UnitPDFMarginal()],
        theta=3.0,
    )

    weights = tm._weight(np.array([[0.25], [0.75]]))

    assert weights.shape == (2,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)


def test_frank_rejects_large_negative_theta_when_exponential_overflows():
    with np.errstate(over="ignore"):
        with pytest.raises(ParameterError, match="too close to 0 or too large"):
            FrankCopula(
                DigitalNetB2(2, seed=127),
                marginals=[stats.uniform(), stats.uniform()],
                theta=-1000.0,
            )
