""" Definition for Lattice, a concrete implementation of DiscreteDistribution """

from ._discrete_distribution import DiscreteDistribution
from .mps_refactor import LatticeSeq
from .._util import DistributionGenerationError, DistributionGenerationWarnings

from numpy import array, log2, vstack, repeat
from numpy.random import Generator, PCG64
import warnings


class Lattice(DiscreteDistribution):
    """ Quasi-Random Lattice low discrepancy sequence (Base 2) """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """
        self.mimics = 'StdUniform'
        self.rng_seed = rng_seed
        self.n_min = 0
        super().__init__()

    def gen_dd_samples(self, replications, n_samples, dimensions, scramble=True):
        """
        Generate r nxd Lattice samples

        Args:
            replications (int): Number of nxd matrices to generate (sample.size()[0])
            n_samples (int): Number of observations (sample.size()[1])
            dimensions (int): Number of dimensions (sample.size()[2])
            scramble (bool): If true, random numbers are in unit cube, otherwise they are non-negative integers

        Returns:
            replications x n_samples x dimensions (numpy array)
        """
        m = log2(n_samples)
        if m % 1 != 0:
            raise DistributionGenerationError("n_samples must be a power of 2")
        m = int(m)
        r = int(replications)
        d = int(dimensions)
        if not hasattr(self, 'lattice_rng'):  # initialize lattice rng and shifts
            self.d = d
            self.r = r
            self.rng = Generator(PCG64(self.rng_seed))
            self.lattice_rng = LatticeSeq(s=self.d)
            self.shifts = self.rng.uniform(0, 1, (self.r, self.d))
        else:
            if d != self.d or r != self.r:
                warnings.warn('''
                    Using dimensions = %d and replications = %d
                    as previously set for this generator.'''
                              % (self.d, self.r),
                              DistributionGenerationWarnings)
        if self.n_min == 0:
            # generate first 2^m points
            x = vstack([self.lattice_rng.calc_block(i) for i in range(m + 1)])
            self.n_min = 2**m
        elif n_samples != self.n_min:
            raise DistributionGenerationError('''
                This Lattice generator has returned a total of %d samples.
                n_samples is expected to be %d
                ''' % (int(self.n_min), int(self.n_min)))
        else:
            # generate self.n_min more samples
            x = self.lattice_rng.calc_block(m + 1)
            self.n_min = 2**(m + 1)
        if scramble:
            # random shift
            x = array([(x + shift_r) % 1 for shift_r in self.shifts])
        else:
            # duplicate unshifted samples
            x = repeat(x[None, :, :], self.r, axis=0)
        return x

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        attributes = ['mimics', 'rng_seed']
        return super().__repr__(attributes)
