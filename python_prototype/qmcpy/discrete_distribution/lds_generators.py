""" This module implements mutiple subclasses of DiscreteDistribution. """


from ._discrete_distribution import DiscreteDistribution
from .mps_refactor import DigitalSeq, LatticeSeq
from .._util import DistributionGenerationError, DistributionGenerationWarnings

from numpy import array, int64, zeros, log2, vstack
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
        self.rng = Generator(PCG64(rng_seed))
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
            x = array([(x + shift_r) % 1 for shift_r in self.shifts])  # random shift
        elif r != 1:  # and scramble==False
            warnings.warn('as scramble==False, setting replications = 1 ',
                          DistributionGenerationWarnings)
        return x


class Sobol(DiscreteDistribution):
    """ Quasi-Random Sobol low discrepancy sequence (Base 2) """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """
        self.mimics = 'StdUniform'
        self.rng = Generator(PCG64(rng_seed))
        super().__init__()

    def gen_dd_samples(self, replications, n_samples, dimensions, scramble=True):
        """
        Generate r nxd Sobol samples

        Args:
            replications (int): Number of nxd matrices to generate (sample.size()[0])
            n_samples (int): Number of observations (sample.size()[1])
            dimensions (int): Number of dimensions (sample.size()[2])
            scramble (bool): If true, random numbers are in unit cube, otherwise they are non-negative integers

        Returns:
            replications x n_samples x dimensions (numpy array)
        """
        if (log2(n_samples)) % 1 != 0:
            raise Exception("n_samples must be a power of 2")
        r = int(replications)
        n = int(n_samples)
        d = int(dimensions)
        if not hasattr(self, 'sobol_rng'):
            self.d = d
            self.r = r
            self.sobol_rng = DigitalSeq(Cs="sobol_Cs.col", m=30, s=d, returnDeepCopy=False)
            self.t = max(32, self.sobol_rng.t)  # we guarantee a depth of >=32 bits for shift
            self.ct = max(0, self.t - self.sobol_rng.t)  # correction factor to scale the integers
            self.shifts = self.rng.integers(0, 2 ** self.t, (self.r, self.d), dtype=int64)
        else:
            if d != self.d or r != self.r:
                warnings.warn('''
                    Using dimensions = %d and replications = %d
                    as previously set for this generator.'''
                    % (self.d, self.r),
                    DistributionGenerationWarnings)
        x = zeros((n, d), dtype=int64)
        for i in range(n):
            next(self.sobol_rng)
            x[i, :] = self.sobol_rng.cur  # set each nxm
        if scramble:
            x = array([(shift_r ^ (x * 2 ** self.ct)) / 2. ** self.t for shift_r in self.shifts])
            # randomly scramble and x contains values in [0, 1]
        elif r != 1:  # and scramble==False
            warnings.warn('as scramble==False, setting replications = 1 ',
                          DistributionGenerationWarnings)
        return x
