""" Definition for Lattice, a concrete implementation of DiscreteDistribution """

from .._discrete_distribution import DiscreteDistribution
from .gail_lattice import gail_lattice_gen
from .mps_lattice import mps_lattice_gen
from ...util import ParameterError
from numpy import array, log2, repeat, vstack, random
import warnings


class Lattice(DiscreteDistribution):
    """ Quasi-Random Lattice low discrepancy sequence (Base 2) """

    parameters = ['dimension','scramble','seed','backend','mimics']
    
    def __init__(self, dimension=1, scramble=True, seed=None, backend='GAIL'):
        """
        Args:
            dimension (int): dimension of samples
            scramble (bool): If True, apply unique scramble to each replication     
            seed (int): seed the random number generator for reproducibility
            backend (str): backend generator
        """
        self.dimension = dimension
        self.scramble = scramble
        self.seed = seed
        self.backend = backend.lower()            
        if self.backend == 'gail':
            self.backend_gen = gail_lattice_gen
        elif self.backend == 'mps':
            self.backend_gen = mps_lattice_gen
        else:
            raise ParameterError("Lattice backend must 'GAIL' or 'MPS'")
        self.mimics = 'StdUniform'
        self.reseed(self.seed)
        super().__init__()

    def gen_samples(self, n=None, n_min=0, n_max=8):
        """
        Generate (n_max-n_min) x self.dimension Lattice samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples.
                     Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence. Must be 0 or n_max/2
            n_max (int): Final index of sequence. Must be a power of 2.

        Returns:
            (n_max-n_min) x self.dimension (ndarray)
        """
        if n:
            n_min = 0
            n_max = n     
        if log2(n_max) % 1 != 0:
            raise ParameterError("n_max must be a power of 2")
        if not (n_min == 0 or n_min == n_max/2):
            raise ParameterError("n_min must be 0 or n_max/2")
        x_lat = self.backend_gen(n_min,n_max,self.dimension)
        if self.scramble: # apply random shift to samples
            x_lat = (x_lat + self.shift)%1
        return x_lat
    
    def reseed(self, new_seed):
        """
        Reseed the generator to get a new scrambling. 
        Args:
            new_seed (int): new seed for generator
        """
        self.seed = new_seed
        random.seed(self.seed)
        self.shift = random.rand(self.dimension)
