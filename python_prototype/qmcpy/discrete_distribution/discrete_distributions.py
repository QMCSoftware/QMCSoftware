from numpy import array, int64, log, random, arange, zeros
from numpy.random import Generator,PCG64

from . import DiscreteDistribution
from .digital_seq import DigitalSeq
from qmcpy.third_party.magic_point_shop import LatticeSeq

class IIDStdUniform(DiscreteDistribution):
    """ IID Standard Uniform Measure """

    def __init__(self, rng_seed=None):
        super().__init__(mimics='StdUniform')
        self.rng = Generator(PCG64(rng_seed))
    
    def gen_samples(self, j, n, m):
        return self.rng.uniform(0, 1, (j,n,m))

class IIDStdGaussian(DiscreteDistribution):
    """ Standard Gaussian Measure """

    def __init__(self, rng_seed=None):
        super().__init__(mimics='StdGaussian')
        self.rng = Generator(PCG64(rng_seed))

    def gen_samples(self, j, n, m):
        return self.rng.standard_normal((j,n,m))

class Lattice(DiscreteDistribution):
    """ Lattice (Base 2) Measure """

    def __init__(self, rng_seed=None):
        super().__init__(mimics='StdUniform')
        self.rng = Generator(PCG64(rng_seed))

    def gen_samples(self, j, n, m):
        x = array([row for row in LatticeSeq(m=int(log(n) / log(2)), s=m)])
        # generate jxnxm data
        shifts = random.rand(j, m)
        x_rs = array([(x + self.rng.uniform(0,1,m)) % 1 for shift in shifts])
        # randomly shift each nxm sample
        return x_rs

class Sobol(DiscreteDistribution):
    """ Sobol (Base 2) Measure """

    def __init__(self, rng_seed=None):
        super().__init__(mimics='StdUniform')
        self.rng = Generator(PCG64(rng_seed))

    def gen_samples(self, j, n, m):
        gen = DigitalSeq(Cs='sobol_Cs.col', m=int(log(n) / log(2)), s=m)
        t = max(32, gen.t)  # we guarantee a depth of >=32 bits for shift
        ct = max(0, t - gen.t)  # correction factor to scale the integers
        shifts = self.rng.integers(0, 2 ** t, (j,m), dtype=int64)
        # generate random shift
        x = zeros((n, m), dtype=int64)
        for i, row in enumerate(gen):
            x[i, :] = gen.cur  # set each nxm
        x_rs = array([(shift ^ (x * 2 ** ct)) / 2. ** t for shift in
                      shifts])  # randomly shift each nxm sample
        return x_rs
