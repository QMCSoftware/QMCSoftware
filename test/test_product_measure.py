import numpy as np
import pytest
import scipy.stats as stats

from qmcpy.discrete_distribution import DigitalNetB2, DummySampler
from qmcpy.true_measure import Gaussian, ProductMeasure, SciPyWrapper, Uniform
from qmcpy.true_measure import ZeroInflatedExpUniform
from qmcpy.util import DimensionError, ParameterError


def test_product_measure_zero_inflated_with_scipy_uniform_shape():
    n = 32
    marginals = [
        ZeroInflatedExpUniform(DummySampler(1), p_zero=0.4, lam=1.5),
        SciPyWrapper(DummySampler(1), stats.uniform(loc=2.0, scale=3.0)),
    ]
    tm = ProductMeasure(sampler=DigitalNetB2(2, seed=23), marginals=marginals)

    x = tm(n)

    assert x.shape == (n, 2)
    assert np.any(x[:, 0] == 0.0)
    assert np.all((2.0 <= x[:, 1]) & (x[:, 1] <= 5.0))


def test_product_measure_replication_shape():
    n = 16
    r = 3
    marginals = [
        ZeroInflatedExpUniform(DummySampler(1), p_zero=0.4, lam=1.5),
        Uniform(DummySampler(1), lower_bound=2.0, upper_bound=5.0),
    ]
    tm = ProductMeasure(
        sampler=DigitalNetB2(2, seed=23, replications=r),
        marginals=marginals,
    )

    x = tm(n)

    assert x.shape == (r, n, 2)


def test_product_measure_marginals_with_different_dimensions():
    n = 32
    marginals = [
        Gaussian(
            DummySampler(2),
            mean=[1.0, -1.0],
            covariance=[[2.0, 0.25], [0.25, 1.0]],
        ),
        ZeroInflatedExpUniform(DummySampler(1), p_zero=0.4, lam=1.5),
    ]
    tm = ProductMeasure(sampler=DigitalNetB2(3, seed=31), marginals=marginals)

    x = tm(n)

    assert x.shape == (n, 3)
    assert np.all(np.isfinite(x[:, :2]))
    assert np.all(x[:, 2] >= 0.0)


def test_product_measure_block_split_and_weight_product():
    n = 16
    marginals = [
        Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0),
        Uniform(
            DummySampler(2),
            lower_bound=[20.0, 30.0],
            upper_bound=[24.0, 36.0],
        ),
    ]
    tm = ProductMeasure(sampler=DigitalNetB2(3, seed=41), marginals=marginals)

    u = tm.discrete_distrib.gen_samples(n)
    x = tm._transform(u)
    x_call, jac = tm(n, return_weights=True)
    expected = np.concatenate(
        [
            marginals[0]._jacobian_transform_r(u[..., :1], return_weights=False),
            marginals[1]._jacobian_transform_r(u[..., 1:], return_weights=False),
        ],
        axis=-1,
    )

    assert x.shape == (n, 3)
    assert np.allclose(x, expected)
    assert np.all((10.0 <= x[:, 0]) & (x[:, 0] <= 12.0))
    assert np.all((20.0 <= x[:, 1]) & (x[:, 1] <= 24.0))
    assert np.all((30.0 <= x[:, 2]) & (x[:, 2] <= 36.0))
    assert np.allclose(tm._weight(x), 1.0 / (2.0 * 4.0 * 6.0))
    assert x_call.shape == (n, 3)
    assert np.allclose(jac, 2.0 * 4.0 * 6.0)


def test_product_measure_invalid_inputs():
    with pytest.raises(ParameterError, match="nonempty list of marginals"):
        ProductMeasure(sampler=DigitalNetB2(1, seed=7), marginals=[])

    with pytest.raises(ParameterError, match="marginal"):
        ProductMeasure(sampler=DigitalNetB2(1, seed=7), marginals=[object()])

    marginals = [Uniform(DummySampler(1))]
    with pytest.raises(DimensionError, match="sum of marginal dimensions"):
        ProductMeasure(sampler=DigitalNetB2(2, seed=7), marginals=marginals)


def test_product_measure_spawn_preserves_marginal_blocks():
    marginals = [
        Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0),
        Uniform(
            DummySampler(2),
            lower_bound=[20.0, 30.0],
            upper_bound=[24.0, 36.0],
        ),
    ]
    tm = ProductMeasure(sampler=DigitalNetB2(3, seed=41), marginals=marginals)

    spawn = tm.spawn(s=1)[0]

    assert isinstance(spawn, ProductMeasure)
    assert spawn.d == 3
    assert spawn.marginals == tm.marginals
    assert np.array_equal(spawn.marginal_dimensions, np.array([1, 2]))

    with pytest.raises(DimensionError):
        tm.spawn(s=1, dimensions=4)


def test_product_measure_does_not_use_empty_marginal_dummy_samples():
    marginals = [
        Uniform(DummySampler(1), lower_bound=0.0, upper_bound=2.0),
        Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0),
    ]

    dummy_samples = marginals[0].discrete_distrib(4)
    assert dummy_samples.shape == (0, 1)

    tm = ProductMeasure(sampler=DigitalNetB2(2, seed=19), marginals=marginals)
    x = tm(8)

    assert x.shape == (8, 2)
    assert np.all((0.0 <= x[:, 0]) & (x[:, 0] <= 2.0))
    assert np.all((10.0 <= x[:, 1]) & (x[:, 1] <= 12.0))


def test_product_measure_replication_means_close_to_uniform_targets():
    n = 1024
    r = 4
    marginals = [
        Uniform(DummySampler(1), lower_bound=0.0, upper_bound=2.0),
        Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0),
    ]
    tm = ProductMeasure(
        sampler=DigitalNetB2(2, seed=101, replications=r),
        marginals=marginals,
    )

    x = tm(n)
    replication_means = x.mean(axis=1)

    assert x.shape == (r, n, 2)
    assert np.allclose(replication_means[:, 0], 1.0, atol=0.03)
    assert np.allclose(replication_means[:, 1], 11.0, atol=0.03)
