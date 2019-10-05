"""
Definitions for IIDDistribution, a DiscreteDistribution
"""
from numpy import array, int64, log, random
from numpy import zeros

from . import DiscreteDistribution
from .magic_point_shop import DigitalSeq,LatticeSeq

class QuasiRandom(DiscreteDistribution):

    def __init__(self, true_distribution=None, seed_rng=None):
        """
        Args:
            accepted_measures (list of strings): Measure objects compatible \
                with the DiscreteDistribution
            seed_rng (int): seed for random number generator to ensure \
                reproduciblity
        """
        accepted_measures = ['Lattice','Sobol'] # Implemented QuasiRandom generators
        if seed_rng: random.seed(seed_rng) # numpy.random for underlying generation
        super().__init__(accepted_measures, true_distribution)

    def gen_distrib(self, n, m, j=16):
        """
        Generate j nxm samples from the quasi-random distribution.
        Randomly shifts each of the j samples.

        Args:
            n (int): Number of observations (sample.size()[1])
            m (int): Number of dimensions (sample.size()[2])
            j (int): Number of nxm matricies to generate (sample.size()[0])

        Returns:
            jxnxm (numpy array)
        """
        if type(self.true_distribution).__name__=='Lattice':
            x = array([row for row in LatticeSeq(m=int(log(n) / log(2)), s=m)]) # generate jxnxm data
            shifts = random.rand(j, m)
            RS_x = array([(x + random.rand(m)) % 1 for shift in shifts]) # randomly shift each nxm sample
            return RS_x
        elif type(self.true_distribution).__name__=='Sobol':
            gen = DigitalSeq(Cs='sobol_Cs.col', m=int(log(n) / log(2)), s=m)
            t = max(32,gen.t) # we will guarantee at least a depth of 32 bits for the shift
            ct = max(0,t-gen.t) # this is the correction factor to scale the integers
            shifts = random.randint(2**t, size=(j,m), dtype=int64) # generate random shift
            x = zeros((n,m),dtype=int64)
            for i,row in enumerate(gen):
                x[i,:] = gen.cur # set each nxm
            x_RS = array([(shift ^ (x * 2**ct)) / 2.**t for shift in shifts]) # randomly shift each nxm sample
            return x_RS
