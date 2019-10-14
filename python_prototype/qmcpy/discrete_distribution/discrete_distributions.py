""" This module implements mutiple subclasses of DiscreteDistribution.
"""
from numpy import array, int64, log, random, zeros

from qmcpy.third_party.magic_point_shop import LatticeSeq
from . import DiscreteDistribution

class IIDStdUniform(DiscreteDistribution):
    """ IID Standard Uniform Measure """

    def __init__(self):
        super().__init__(mimics='StdUniform')

    def gen_samples(self, j, n, m):
        return random.rand(j, n, m)

class IIDStdGaussian(DiscreteDistribution):
    """ Standard Gaussian Measure """

    def __init__(self):
        super().__init__(mimics='StdGaussian')

    def gen_samples(self, j, n, m):
        return random.randn(j, n, m)

class Lattice(DiscreteDistribution):
    """ Lattice (Base 2) Measure """

    def __init__(self):
        super().__init__(mimics='StdUniform')

    def gen_samples(self, j, n, m):
        x = array([row for row in LatticeSeq(m=int(log(n) / log(2)), s=m)])
        # generate jxnxm data
        shifts = random.rand(j, m)
        x_rs = array([(x + random.rand(m)) % 1 for shift in shifts])
        # randomly shift each nxm sample
        return x_rs

class Sobol(DiscreteDistribution):
    """ Sobol (Base 2) Measure """

    def __init__(self):
        super().__init__(mimics='StdUniform')

    def gen_samples(self, j, n, m):
        gen = DigitalSeq(Cs='sobol_Cs.col', m=int(log(n) / log(2)), s=m)
        t = max(32, gen.t)  # we guarantee a depth of >=32 bits for shift
        ct = max(0, t - gen.t)  # correction factor to scale the integers
        shifts = random.randint(2 ** t, size=(j, m), dtype=int64)
        # generate random shift
        x = zeros((n, m), dtype=int64)
        for i, _ in enumerate(gen):
            x[i, :] = gen.cur  # set each nxm
        x_rs = array([(shift ^ (x * 2 ** ct)) / 2. ** t for shift in
                      shifts])  # randomly shift each nxm sample
        return x_rs

# API
from .digital_seq import DigitalSeq
