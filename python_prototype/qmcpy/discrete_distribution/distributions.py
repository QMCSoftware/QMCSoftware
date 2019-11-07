""" This module implements mutiple subclasses of DiscreteDistribution. """

from numpy import array, int64, log, random, zeros
from numpy.random import Generator, PCG64

from ._discrete_distribution import DiscreteDistribution
from ..third_party.magic_point_shop import LatticeSeq
from .digital_seq import DigitalSeq


class IIDStdUniform(DiscreteDistribution):
    """ IID Standard Uniform """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """
        super().__init__(mimics="StdUniform", rng_seed=rng_seed)
        self.rng = Generator(PCG64(rng_seed))

    def gen_dd_samples(self, r, n, d):
        """
        Generate r nxd IID Standard Uniform samples

        Args:
            r (int): Number of nxd matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            d (int): Number of dimensions (sample.size()[2])

        Returns:
            rxnxd (numpy array)
        """
        return self.rng.uniform(0, 1, (int(r), int(n), int(d)))


class IIDStdGaussian(DiscreteDistribution):
    """ Standard Gaussian """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """
        super().__init__(mimics="StdGaussian", rng_seed=rng_seed)
        self.rng = Generator(PCG64(rng_seed))

    def gen_dd_samples(self, r, n, d):
        """
        Generate r nxd IID Standard Gaussian samples

        Args:
            r (int): Number of nxd matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            d (int): Number of dimensions (sample.size()[2])

        Returns:
            rxnxd (numpy array)
        """
        return self.rng.standard_normal((int(r), int(n), int(d)))


class Lattice(DiscreteDistribution):
    """ Quasi-Random Lattice low discrepancy sequence (Base 2) """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """
        super().__init__(mimics="StdUniform", rng_seed=rng_seed)
        self.rng = Generator(PCG64(rng_seed))

    def gen_dd_samples(self, r, n, d):
        """
        Generate r nxd Lattice samples

        Args:
            r (int): Number of nxd matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            d (int): Number of dimensions (sample.size()[2])

        Returns:
            rxnxd (numpy array)
        """
        if not hasattr(self,'lattice_rng'): # initialize lattice rng and shifts
            self.lattice_rng = LatticeSeq(m=20, s=int(d))
            self.shifts = self.rng.uniform(0,1,(int(r),int(d)))
        x = array([next(self.lattice_rng) for i in range(int(n))])
        x_rs = array([(x + shift_r) % 1 for shift_r in self.shifts]) # random shift
        return x_rs


class Sobol(DiscreteDistribution):
    """ Quasi-Random Sobol low discrepancy sequence (Base 2) """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """
        super().__init__(mimics="StdUniform", rng_seed=rng_seed)
        self.rng = Generator(PCG64(rng_seed))

    def gen_dd_samples(self, r, n, d):
        """
        Generate r nxd Sobol samples

        Args:
            r (int): Number of nxd matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            d (int): Number of dimensions (sample.size()[2])

        Returns:
            rxnxd (numpy array)
        """
        if not hasattr(self,'sobol_rng'):
            self.sobol_rng = DigitalSeq(Cs="sobol_Cs.col", m=20, s=int(d))
            self.t = max(32 , self.sobol_rng.t) # we guarantee a depth of >=32 bits for shift
            self.ct = max(0, self.t-self.sobol_rng.t)  # correction factor to scale the integers
            self.shifts = self.rng.integers(0, 2 ** self.t, (int(r), int(d)), dtype=int64)
        x = zeros((int(n), int(d)), dtype=int64)
        for i in range(int(n)):
            next(self.sobol_rng)
            x[i, :] = self.sobol_rng.cur  # set each nxm
        x_rs = array([(shift_r ^ (x * 2 ** self.ct)) / 2. ** self.t for shift_r in self.shifts])
        #   randomly scramble
        return x_rs
