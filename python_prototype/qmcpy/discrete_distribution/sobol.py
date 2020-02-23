""" Definition for Sobol, a concrete implementation of DiscreteDistribution """

from numpy import array, int64, log2, random, repeat, zeros
import numpy.random as npr
from torch.quasirandom import SobolEngine
import warnings

from ._discrete_distribution import DiscreteDistribution
from .mps_refactor import DigitalSeq
from ..util import DistributionGenerationWarnings, ParameterError


class Sobol(DiscreteDistribution):
    """ Quasi-Random Sobol low discrepancy sequence (Base 2) """

    def __init__(self, rng_seed=None, backend='mps'):
        """
        Initialize Sobol sequence generator.

        Args:
            rng_seed (int): seed the random number generator for reproducibility

        """
        self.mimics = 'StdUniform'
        self.rng_seed = rng_seed
        self.backend = backend.lower()
        if self.backend not in ['mps', 'pytorch']:
            raise ParameterError("Sobol backend must be either 'mps' or 'pytorch'")
        if self.backend == 'pytorch':
            raise Exception("PyTorch SobolEngine issue. " +
                "See https://github.com/pytorch/pytorch/issues/32047")
        super().__init__()

    def gen_dd_samples(self, replications, n_samples, dimensions, scramble=True):
        """
        Generate :math:`r` samples of :math:`n \cdot d` Sobol points.

        Args:
            replications (int): :math:`r`, number of nxd matrices to generate
            n_samples (int): :math:`d`, number of observations
            dimensions (int): :math:`d`, number of dimensions
            scramble (bool): If true, random numbers are in unit cube, otherwise they are non-negative integers

        Returns:
            x (numpy.array) of with dimensions :math:`r \cdot n \cdot d`

        """
        if (log2(n_samples)) % 1 != 0:
            raise Exception("n_samples must be a power of 2")
        r = int(replications)
        n = int(n_samples)
        d = int(dimensions)
        if not hasattr(self, 'sobol_rng'):
            # Initialize the backend Sobol Generator on first call
            # Permanently set the dimension and number of replications
            self.d = d
            self.r = r
            if self.backend == 'mps':
                self.rng = random.Generator(random.PCG64(self.rng_seed))
                self.sobol_rng = DigitalSeq(Cs="sobol_Cs.col", m=30, s=self.d)
                # we guarantee a depth of >=32 bits for shift
                self.t = max(32, self.sobol_rng.t)
                # correction factor to scale the integers
                self.ct = max(0, self.t - self.sobol_rng.t)
                self.shifts = self.rng.integers(
                    0, 2 ** self.t, (self.r, self.d), dtype=int64)
            elif self.backend == 'pytorch':
                # Initialize self.r SobolEngines
                temp_seed = random.randint(100) if self.rng_seed is None else self.rng_seed
                self.sobol_rng = [SobolEngine(dimension=self.d, scramble=scramble, seed=seed)
                                  for seed in range(temp_seed, temp_seed + self.r)]
        else:
            # Not the first call to this method
            if d != self.d or r != self.r:
                warnings.warn('''
                    Using dimensions = %d and replications = %d
                    as previously set for this generator.'''
                              % (self.d, self.r),
                              DistributionGenerationWarnings)
        if self.backend == 'mps':
            x = zeros((n, d), dtype=int64)
            for i in range(n):
                next(self.sobol_rng)
                x[i, :] = self.sobol_rng.cur  # set each nxm
            if scramble:
                x = array([(shift_r ^ (x * 2 ** self.ct)) / 2. **
                           self.t for shift_r in self.shifts])
                # randomly scramble and x contains values in [0, 1]
            else:
                # duplicate unshifted samples
                x = repeat(x[None, :, :], self.r, axis=0)
        elif self.backend == 'pytorch':
            x = array([self.sobol_rng[i].draw(n).numpy()
                       for i in range(self.r)])
        return x

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args:
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        attributes = ['mimics', 'rng_seed', 'backend']
        return super().__repr__(attributes)
