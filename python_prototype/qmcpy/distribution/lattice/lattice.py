""" Definition for Lattice, a concrete implementation of Distribution """

from .._distribution import Distribution
from .gail_lattice import gail_lattice_gen
from .mps_lattice import mps_lattice_gen
from ...util import DistributionGenerationError, DistributionGenerationWarnings, ParameterError
from numpy import array, log2, repeat, vstack
from numpy.random import Generator, PCG64
import warnings


class Lattice(Distribution):
    """ Quasi-Random Lattice low discrepancy sequence (Base 2) """

    def __init__(self, dimension=1, scramble=False, replications=0, seed=None, backend='GAIL'):
        """
        Args:
            dimension (int): dimension of samples
            scramble (bool): If True, apply unique scramble to each replication
            replications (int): Number of nxd matrices to generate
                replications set to 0 ignores replications and returns (n_max-n_min)xd samples            
            seed (int): seed the random number generator for reproducibility
            backend (str): backend generator
        """
        self.dimension = dimension
        self.scramble = scramble
        self.squeeze = (replications==0)
        self.replications = max(1,replications)
        self.seed = seed
        self.shifts = Generator(PCG64(self.seed)).uniform(0, 1, (self.replications, self.dimension))
        self.backend = backend.lower()            
        if self.backend == 'gail':
            self.backend_gen = gail_lattice_gen
        elif self.backend == 'mps':
            self.backend_gen = mps_lattice_gen
        else:
            raise ParameterError("Lattice backend must 'GAIL' or 'MPS'")
        self.mimics = 'StdUniform'
        
        super().__init__()

    def gen_samples(self, n_min=0, n_max=8):
        """
        Generate self.replications (n_max-n_min)xself.d Lattice samples

        Args:
            n_min (int): Starting index of sequence. Must be 0 or n_max/2
            n_max (int): Final index of sequence. Must be a power of 2. 

        Returns:
            self.replications x (n_max-n_min) x self.dimension (ndarray)
        """
        if log2(n_max) % 1 != 0:
            raise DistributionGenerationError("n_max must be a power of 2")
        if not (n_min == 0 or n_min == n_max/2):
            raise DistributionGenerationError("n_min must be 0 or n_max/2")
        x_lat = self.backend_gen(n_min,n_max,self.dimension)
        if self.scramble: # apply random shift to samples
            x_lat_reps = array([(x_lat + shift_r) % 1 for shift_r in self.shifts])
        else: # duplicate unshifted samples
            x_lat_reps = repeat(x_lat[None, :, :], self.replications, axis=0)
        if self.squeeze:
            x_lat_reps = x_lat_reps.squeeze(0)
        return x_lat_reps

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args:
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        attributes = ['dimension','scramble','replications','seed','backend','mimics']
        return super().__repr__(attributes)
