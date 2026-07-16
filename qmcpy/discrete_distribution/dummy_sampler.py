import numpy as np

from .abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ..util import ParameterError


class DummySampler(AbstractLDDiscreteDistribution):
    r"""
    Placeholder discrete distribution for constructing true-measure marginals.

   ``DummySampler`` is intended primarily for ``ProductMeasure`` marginals.
    It allows true measures to be constructed without attaching a real
    sampling algorithm. During ProductMeasure sampling, only the outer
    sampler generates QMC points. The attached DummySamplers are never
    sampled directly.

    ``DummySampler`` intentionally generates no sample points. Direct sampling
    returns an empty array with the usual replication-aware shape.

    Examples
    --------
    >>> from qmcpy.discrete_distribution import DummySampler
    >>> sampler = DummySampler(2)
    >>> sampler.d
    2
    >>> sampler.replications
    1
    >>> sampler(4).shape
    (0, 2)
    >>> DummySampler(2, replications=3)(4).shape
    (3, 0, 2)
    """

    def __init__(self, dimension=1, replications=None, seed=None, warn=True):
        # DummySampler exists only to satisfy the current AbstractTrueMeasure API.
        # ProductMeasure never samples from these dummy samplers. They only provide
        # a placeholder discrete distribution while the outer ProductMeasure sampler
        # generates the actual QMC points.

        # DummySampler does not expose any configurable parameters beyond those
        # required by the AbstractLDDiscreteDistribution interface.
        self.parameters = []

        # ProductMeasure still transforms points from the unit cube, so this dummy
        # sampler mimics the standard uniform distribution even though it never
        # generates any sample points
        self.mimics = "StdUniform"

        # Initialize the common discrete-distribution infrastructure
        # (dimension, replications, random seed, spawning support, etc.)
        super(DummySampler, self).__init__(
            dimension,
            replications,
            seed,
            d_limit=np.inf,
            n_limit=np.inf,
        )

    def gen_samples(
        self, n=None, n_min=None, n_max=None, return_binary=False, warn=True
    ):
        ## The inherited gen_samples() implementation validates that _gen_samples()
        # returns exactly the requested number of points. DummySampler intentionally
        # generates zero points, so we override only the public wrapper while keeping
        # the same argument parsing and validation logic. The only difference is that
        # an empty sample axis is accepted.

        # Parse the supported QMCPy calling conventions:
        #   sampler(n)
        #   sampler(n_min, n_max)
        #   sampler(n, n_min)   
        if n is not None and n_min is None and n_max is None:
            n_min = 0
            n_max = int(n)
        elif n is None and n_min is not None and n_max is not None:
            n_min = int(n_min)
            n_max = int(n_max)
        elif n is not None and n_min is not None and n_max is None:
            n_max = int(n_min)
            n_min = int(n)
        else:
            raise ParameterError("Please provide either n or (n_min,n_max)")

        if not (0 <= n_min and n_min <= n_max and n_max <= self.n_limit):
            raise ParameterError(
                "require 0 <= n_min (%d) <= n_max (%d) <= n_limit (%d)"
                % (n_min, n_max, self.n_limit)
            )

        x = self._gen_samples(
            n_min=n_min, n_max=n_max, return_binary=return_binary, warn=warn
        )

        # Match the standard QMCPy output convention:
        #   no replications  -> (0, d)
        #   with replications -> (r, 0, d)
        return x[0] if self.no_replications else x

    def _gen_samples(self, n_min, n_max, return_binary, warn):
        # DummySampler intentionally produces no sample points.
        # ProductMeasure samples only from its outer sampler, so the dummy sampler
        # simply returns an empty replication-aware array.

        # Binary output has no meaningful interpretation for an empty sampler.
        if return_binary:
            raise ParameterError("DummySampler does not support return_binary=True")
        
        # Shape follows the internal AbstractLDDiscreteDistribution convention:
        # (replications, n_points, dimension). Here n_points is always zero.
        return np.empty((self.replications, 0, self.d))

    def _spawn(self, child_seed, dimension):
        # Spawning preserves the DummySampler behavior while updating the
        # dimension and seed, following the standard QMCPy spawning interface.
        # Replication settings are preserved automatically.
        return DummySampler(
            dimension=dimension,
            replications=None if self.no_replications else self.replications,
            seed=child_seed,
            warn=False,
        )
