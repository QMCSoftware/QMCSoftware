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
    
    def gen_samples(self, j, n, m):
        """
        Generate j nxm IID Standard Uniform samples

        Args:
            j (int): Number of nxm matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            m (int): Number of dimensions (sample.size()[2])

        Returns:
            jxnxm (numpy array)
        """
        return self.rng.uniform(0, 1, (j,n,m))

class IIDStdGaussian(DiscreteDistribution):
    """ Standard Gaussian """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """ 
        super().__init__(mimics='StdGaussian')
        self.rng = Generator(PCG64(rng_seed))

    def gen_samples(self, j, n, m):
        """
        Generate j nxm IID Standard Gaussian samples

        Args:
            j (int): Number of nxm matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            m (int): Number of dimensions (sample.size()[2])

        Returns:
            jxnxm (numpy array)
        """
        return self.rng.standard_normal((j,n,m))

class Lattice(DiscreteDistribution):
    """ Quasi-Random Lattice low discrepancy sequence (Base 2) """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """ 
        super().__init__(mimics='StdUniform')
        self.rng = Generator(PCG64(rng_seed))

    def gen_samples(self, j, n, m):
        """
        Generate j nxm Lattice samples

        Args:
            j (int): Number of nxm matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            m (int): Number of dimensions (sample.size()[2])

        Returns:
            jxnxm (numpy array)
        """
        x = array([row for row in LatticeSeq(m=int(log(n) / log(2)), s=m)])
        # generate jxnxm data
        shifts = random.rand(j, m)
        x_rs = array([(x + self.rng.uniform(0,1,m)) % 1 for shift in shifts])
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

    def gen_samples(self, j, n, m):
        """
        Generate j nxm Sobol samples

        Args:
            j (int): Number of nxm matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            m (int): Number of dimensions (sample.size()[2])

        Returns:
            jxnxm (numpy array)
        """
        gen = DigitalSeq(Cs='sobol_Cs.col', m=int(log(n) / log(2)), s=m)
        t = max(32, gen.t)  # we guarantee a depth of >=32 bits for shift
        ct = max(0, t - gen.t)  # correction factor to scale the integers
        shifts = self.rng.integers(0, 2 ** t, (j,m), dtype=int64)
        # generate random shift
        x = zeros((n, m), dtype=int64)
        for i, _ in enumerate(gen):
            x[i, :] = gen.cur  # set each nxm
        x_rs = array([(shift ^ (x * 2 ** ct)) / 2. ** t for shift in
                      shifts])  # randomly shift each nxm sample
        return x_rs