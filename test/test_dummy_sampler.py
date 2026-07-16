import numpy as np
import pytest

from qmcpy.discrete_distribution import DummySampler
from qmcpy.util import ParameterError


def test_dummy_sampler_constructs_dimension_one():
    sampler = DummySampler(1)

    assert sampler.d == 1
    assert sampler.replications == 1
    assert sampler.no_replications
    assert sampler.mimics == "StdUniform"
    assert sampler.parameters == []


def test_dummy_sampler_constructs_larger_dimensions():
    sampler = DummySampler(3, seed=7)

    assert sampler.d == 3
    assert sampler.replications == 1
    assert sampler.no_replications
    assert np.array_equal(sampler.dvec, np.arange(3))


def test_dummy_sampler_constructs_larger_dimension_with_replications():
    sampler = DummySampler(4, replications=3, seed=7)

    assert sampler.d == 4
    assert sampler.replications == 3
    assert not sampler.no_replications
    assert np.array_equal(sampler.dvec, np.arange(4))


def test_dummy_sampler_direct_sampling_returns_empty_array():
    sampler = DummySampler(2)

    x = sampler(8)

    assert x.shape == (0, 2)
    assert x.size == 0


def test_dummy_sampler_replicated_direct_sampling_returns_empty_array():
    sampler = DummySampler(2, replications=3)

    x = sampler(8)

    assert x.shape == (3, 0, 2)
    assert x.size == 0


def test_dummy_sampler_supported_calling_conventions_return_empty_arrays():
    sampler = DummySampler(2)

    x_n = sampler(n=4)
    x_range = sampler(n_min=2, n_max=6)
    x_n_to_n_min = sampler(n=2, n_min=6)

    assert x_n.shape == (0, 2)
    assert x_range.shape == (0, 2)
    assert x_n_to_n_min.shape == (0, 2)
    assert x_n.size == x_range.size == x_n_to_n_min.size == 0


def test_dummy_sampler_rejects_return_binary():
    sampler = DummySampler(2)

    with pytest.raises(ParameterError, match="return_binary"):
        sampler(4, return_binary=True)


def test_dummy_sampler_internal_gen_samples_returns_base_shape():
    sampler = DummySampler(2)

    x = sampler._gen_samples(n_min=0, n_max=8, return_binary=False, warn=True)

    assert x.shape == (1, 0, 2)
    assert x.size == 0


def test_dummy_sampler_spawn_preserves_relevant_fields():
    sampler = DummySampler(2, replications=3, seed=11)

    spawned = sampler.spawn(s=2, dimensions=[1, 5])

    assert [spawn.d for spawn in spawned] == [1, 5]
    assert [spawn.replications for spawn in spawned] == [3, 3]
    assert all(isinstance(spawn, DummySampler) for spawn in spawned)


def test_dummy_sampler_spawn_without_explicit_replications():
    sampler = DummySampler(2, seed=11)

    spawned = sampler.spawn(s=1, dimensions=4)[0]

    assert spawned.d == 4
    assert spawned.replications == 1
    assert spawned.no_replications
