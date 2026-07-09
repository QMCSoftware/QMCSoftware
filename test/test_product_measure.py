import numpy as np
import pytest
import scipy.stats as stats

from qmcpy.discrete_distribution import DigitalNetB2
from qmcpy.true_measure import Gaussian, ProductMeasure, SciPyWrapper, Uniform
from qmcpy.true_measure import ZeroInflatedExpUniform
from qmcpy.util import DimensionError, ParameterError


def test_product_measure_zero_inflated_with_scipy_uniform_shape():
    n = 32
    children = [
        ZeroInflatedExpUniform(DigitalNetB2(1, seed=17), p_zero=0.4, lam=1.5),
        SciPyWrapper(DigitalNetB2(1, seed=19), stats.uniform(loc=2.0, scale=3.0)),
    ]
    tm = ProductMeasure(sampler=DigitalNetB2(2, seed=23), children=children)

    x = tm(n)

    assert x.shape == (n, 2)
    assert np.any(x[:, 0] == 0.0)
    assert np.all((2.0 <= x[:, 1]) & (x[:, 1] <= 5.0))


def test_product_measure_replication_shape():
    n = 16
    r = 3
    children = [
        ZeroInflatedExpUniform(DigitalNetB2(1, seed=17), p_zero=0.4, lam=1.5),
        Uniform(DigitalNetB2(1, seed=19), lower_bound=2.0, upper_bound=5.0),
    ]
    tm = ProductMeasure(
        sampler=DigitalNetB2(2, seed=23, replications=r),
        children=children,
    )

    x = tm(n)

    assert x.shape == (r, n, 2)


def test_product_measure_children_with_different_dimensions():
    n = 32
    children = [
        Gaussian(
            DigitalNetB2(2, seed=23),
            mean=[1.0, -1.0],
            covariance=[[2.0, 0.25], [0.25, 1.0]],
        ),
        ZeroInflatedExpUniform(DigitalNetB2(1, seed=29), p_zero=0.4, lam=1.5),
    ]
    tm = ProductMeasure(sampler=DigitalNetB2(3, seed=31), children=children)

    x = tm(n)

    assert x.shape == (n, 3)
    assert np.all(np.isfinite(x[:, :2]))
    assert np.all(x[:, 2] >= 0.0)


def test_product_measure_block_split_and_weight_product():
    n = 16
    children = [
        Uniform(DigitalNetB2(1, seed=31), lower_bound=10.0, upper_bound=12.0),
        Uniform(
            DigitalNetB2(2, seed=37),
            lower_bound=[20.0, 30.0],
            upper_bound=[24.0, 36.0],
        ),
    ]
    tm = ProductMeasure(sampler=DigitalNetB2(3, seed=41), children=children)

    u = tm.discrete_distrib.gen_samples(n)
    x = tm._transform(u)
    x_call, jac = tm(n, return_weights=True)
    expected = np.concatenate(
        [
            children[0]._jacobian_transform_r(u[..., :1], return_weights=False),
            children[1]._jacobian_transform_r(u[..., 1:], return_weights=False),
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
    with pytest.raises(ParameterError):
        ProductMeasure(sampler=DigitalNetB2(1, seed=7), children=[])

    with pytest.raises(ParameterError):
        ProductMeasure(sampler=DigitalNetB2(1, seed=7), children=[object()])

    children = [Uniform(DigitalNetB2(1, seed=31))]
    with pytest.raises(DimensionError):
        ProductMeasure(sampler=DigitalNetB2(2, seed=7), children=children)


def test_product_measure_spawn_preserves_child_blocks():
    children = [
        Uniform(DigitalNetB2(1, seed=31), lower_bound=10.0, upper_bound=12.0),
        Uniform(
            DigitalNetB2(2, seed=37),
            lower_bound=[20.0, 30.0],
            upper_bound=[24.0, 36.0],
        ),
    ]
    tm = ProductMeasure(sampler=DigitalNetB2(3, seed=41), children=children)

    spawn = tm.spawn(s=1)[0]

    assert isinstance(spawn, ProductMeasure)
    assert spawn.d == 3
    assert np.array_equal(spawn.child_dimensions, np.array([1, 2]))

    with pytest.raises(DimensionError):
        tm.spawn(s=1, dimensions=4)
