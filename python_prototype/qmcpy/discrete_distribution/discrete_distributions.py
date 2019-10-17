""" This module implements mutiple subclasses of DiscreteDistribution. """

from numpy import array, int64, log, random, arange, zeros
from numpy.random import Generator,PCG64

from . import DiscreteDistribution
from qmcpy.third_party.magic_point_shop import LatticeSeq
from . import DigitalSeq


class IIDStdUniform(DiscreteDistribution):
    """ IID Standard Uniform """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """ 
        super().__init__(mimics='StdUniform')
        self.rng = Generator(PCG64(rng_seed))
    
    def gen_samples(self, r, n, d):
        """
        Generate r nxd IID Standard Uniform samples

        Args:
            r (int): Number of nxd matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            d (int): Number of dimensions (sample.size()[2])

        Returns:
            rxnxd (numpy array)
        """
        return self.rng.uniform(0, 1, (r,n,d))

class IIDStdGaussian(DiscreteDistribution):
    """ Standard Gaussian """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """ 
        super().__init__(mimics='StdGaussian')
        self.rng = Generator(PCG64(rng_seed))

    def gen_samples(self, r, n, d):
        """
        Generate r nxd IID Standard Gaussian samples

        Args:
            r (int): Number of nxd matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            d (int): Number of dimensions (sample.size()[2])

        Returns:
            rxnxd (numpy array)
        """
        return self.rng.standard_normal((r,n,d))

class Lattice(DiscreteDistribution):
    """ Quasi-Random Lattice low discrepancy sequence (Base 2) """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """ 
        super().__init__(mimics='StdUniform')
        self.rng = Generator(PCG64(rng_seed))

    def gen_samples(self, r, n, d):
        """
        Generate r nxd Lattice samples

        Args:
            r (int): Number of nxd matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            d (int): Number of dimensions (sample.size()[2])

        Returns:
            rxnxd (numpy array)
        """
        x = array([row for row in LatticeSeq(m=int(log(n) / log(2)), s=d)])
        # generate jxnxm data
        shifts = random.rand(r, d)
        x_rs = array([(x + self.rng.uniform(0,1,d)) % 1 for shift in shifts])
        # randomly shift each nxm sample
        return x_rs

class Sobol(DiscreteDistribution):
    """ Quasi-Random Sobol low discrepancy sequence (Base 2) """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """ 
        super().__init__(mimics='StdUniform')
        self.rng = Generator(PCG64(rng_seed))

    def gen_samples(self, r, n, d):
        """
        Generate r nxd Sobol samples

        Args:
            r (int): Number of nxd matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            d (int): Number of dimensions (sample.size()[2])

        Returns:
            rxnxd (numpy array)
        """
        gen = DigitalSeq(Cs='sobol_Cs.col', m=int(log(n) / log(2)), s=d)
        t = max(32, gen.t)  # we guarantee a depth of >=32 bits for shift
        ct = max(0, t - gen.t)  # correction factor to scale the integers
        shifts = self.rng.integers(0, 2 ** t, (r,d), dtype=int64)
        # generate random shift
        x = zeros((n, d), dtype=int64)
        for i, _ in enumerate(gen):
            x[i, :] = gen.cur  # set each nxm
        x_rs = array([(shift ^ (x * 2 ** ct)) / 2. ** t for shift in
                      shifts])  # randomly shift each nxm sample
        return x_rs