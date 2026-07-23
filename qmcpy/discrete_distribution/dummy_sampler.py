import numpy as np

from .abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ..util import ParameterError


class DummySampler(AbstractLDDiscreteDistribution):
    r"""
    Placeholder discrete distribution for constructing true-measure marginals.

    ``DummySampler`` is useful when a true measure is needed only for its
    dimension, transform, range, and weight behavior. QMCPy's current
    ``AbstractTrueMeasure`` interface requires each true measure to be
    constructed with an attached sampler, but ``ProductMeasure`` samples only
    from its own outer sampler.

    Direct calls to ``DummySampler`` return an uninitialized ``np.empty`` array
    with the standard QMCPy sample shape. These placeholder values are not
    meaningful QMC points and should not be inspected.

    Examples
    --------
    >>> from qmcpy.discrete_distribution import DummySampler
    >>> sampler = DummySampler(2)
    >>> sampler.d
    2
    >>> sampler.replications
    1
    >>> sampler(4).shape
    (4, 2)
    >>> DummySampler(2, replications=3)(4).shape
    (3, 4, 2)
    """

    def __init__(self, dimension=1, replications=None, seed=None, warn=True):
        # Keep the same constructor as other discrete distributions.
        del warn

        # DummySampler has no extra parameters.
        self.parameters = []

        # True measures expect unit-cube input.
        self.mimics = "StdUniform"

        # Initialize the common discrete distribution settings.
        super(DummySampler, self).__init__(
            dimension,
            replications,
            seed,
            d_limit=10_000,
            n_limit=2**32,
        )

    def _gen_samples(self, n_min, n_max, return_binary, warn):
        # warn is unused.
        del warn
        if return_binary:
            # DummySampler does not support binary output.
            raise ParameterError("DummySampler does not support return_binary=True")

        # The base class passes either:
        #   sampler(n)           -> n_min=0, n_max=n
        #   sampler(n_min,n_max) -> use the given range
        # Number of requested samples is n_max - n_min.
        n = n_max - n_min

        # Return an array with the requested shape.
        # ProductMeasure only needs the shape.
        return np.empty((self.replications, n, self.d))

    def _spawn(self, child_seed, dimension):
        # Create a new DummySampler with the given dimension and seed.
        # Preserve the current replication setting.
        return DummySampler(
            dimension=dimension,
            replications=None if self.no_replications else self.replications,
            seed=child_seed,
            warn=False,
        )
