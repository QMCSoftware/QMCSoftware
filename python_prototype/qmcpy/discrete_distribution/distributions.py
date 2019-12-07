""" This module implements mutiple subclasses of DiscreteDistribution. """

from ._discrete_distribution import DiscreteDistribution
from ..third_party.magic_point_shop import LatticeSeq
from .digital_seq import DigitalSeq

from numpy import array, int64, log, random, zeros
from numpy.random import Generator, PCG64


class IIDStdUniform(DiscreteDistribution):
    """ IID Standard Uniform """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """
        self.mimics = 'StdUniform'
        self.rng = Generator(PCG64(rng_seed))
        super().__init__()

    def gen_dd_samples(self, replications, n_samples, dimensions):
        """
        Generate r nxd IID Standard Uniform samples

        Args:
            replications (int): Number of nxd matrices to generate (sample.size()[0])
            n_samples (int): Number of observations (sample.size()[1])
            dimensions (int): Number of dimensions (sample.size()[2])

        Returns:
            replications x n_samples x dimensions (numpy array)
        """
        r = int(replications)
        n = int(n_samples)
        d = int(dimensions)
        return self.rng.uniform(0, 1, (r, n, d))


class IIDStdGaussian(DiscreteDistribution):
    """ Standard Gaussian """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """
        self.mimics = 'StdGaussian'
        self.rng = Generator(PCG64(rng_seed))
        super().__init__()

    def gen_dd_samples(self, replications, n_samples, dimensions):
        """
        Generate r nxd IID Standard Gaussian samples

        Args:
            replications (int): Number of nxd matrices to generate (sample.size()[0])
            n_samples (int): Number of observations (sample.size()[1])
            dimensions (int): Number of dimensions (sample.size()[2])

        Returns:
            replications x n_samples x dimensions (numpy array)
        """
        r = int(replications)
        n = int(n_samples)
        d = int(dimensions)
        return self.rng.standard_normal((r, n, d))


class Lattice(DiscreteDistribution):
    """ Quasi-Random Lattice low discrepancy sequence (Base 2) """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """
        self.mimics = 'StdUniform'
        self.rng = Generator(PCG64(rng_seed))
        super().__init__()

    def gen_dd_samples(self, replications, n_samples, dimensions):
        """
        Generate r nxd Lattice samples

        Args:
            replications (int): Number of nxd matrices to generate (sample.size()[0])
            n_samples (int): Number of observations (sample.size()[1])
            dimensions (int): Number of dimensions (sample.size()[2])

        Returns:
            replications x n_samples x dimensions (numpy array)
        """
        r = int(replications)
        n = int(n_samples)
        d = int(dimensions)
        if not hasattr(self, 'lattice_rng'):  # initialize lattice rng and shifts
            self.lattice_rng = LatticeSeq(m=20, s=int(d))
            self.shifts = self.rng.uniform(0, 1, (int(r), int(d)))
        x = array([next(self.lattice_rng) for i in range(int(n))])
        x_rs = array([(x + shift_r) % 1 for shift_r in self.shifts])  # random shift
        return x_rs


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

    def gen_dd_samples(self, replications, n_samples, dimensions, returnDeepCopy=True):
        """
        Generate r nxd Sobol samples

        Args:
            replications (int): Number of nxd matrices to generate (sample.size()[0])
            n_samples (int): Number of observations (sample.size()[1])
            dimensions (int): Number of dimensions (sample.size()[2])

        Returns:
            replications x n_samples x dimensions (numpy array)
        """
        r = int(replications)
        n = int(n_samples)
        d = int(dimensions)
        if not hasattr(self, 'sobol_rng'):
            self.sobol_rng = DigitalSeq(Cs="sobol_Cs.col", m=n, s=d, returnDeepCopy=returnDeepCopy)
            self.t = max(32, self.sobol_rng.t)  # we guarantee a depth of >=32 bits for shift
            self.ct = max(0, self.t - self.sobol_rng.t)  # correction factor to scale the integers
            self.shifts = self.rng.integers(0, 2 ** self.t, (r, d), dtype=int64)
        x = zeros((n, d), dtype=int64)
        for i in range(n):
            next(self.sobol_rng)
            x[i, :] = self.sobol_rng.cur  # set each nxm; x contains non-negative integers
        x_rs = array([(shift_r ^ (x * 2 ** self.ct)) / 2. ** self.t for shift_r in self.shifts])
        #   randomly scramble and x_rs contains values in [0, 1]
        return x_rs
