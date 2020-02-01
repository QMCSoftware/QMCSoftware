""" Definition for Sobol, a concrete implementation of DiscreteDistribution """

from .._discrete_distribution import DiscreteDistribution
from .mps_sobol import DigitalSeq
from ...util import DistributionGenerationWarnings, ParameterError, ParameterWarning
from numpy import array, int64, log2, repeat, zeros
from numpy.random import Generator, PCG64, randint
from torch.quasirandom import SobolEngine
import warnings


class Sobol(DiscreteDistribution):
    """ Quasi-Random Sobol low discrepancy sequence (Base 2) """

    def __init__(self, dimension=1, scramble=False, replications=0, seed=None, backend='MPS'):
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
        self.backend = backend.lower()
        rng = Generator(PCG64(self.seed))
        if self.backend == 'mps': 
            self.sobol_rng = DigitalSeq(m=30, s=self.dimension)
            # we guarantee a depth of >=32 bits for shift
            self.t = max(32, self.sobol_rng.t)
            # correction factor to scale the integers
            self.ct = max(0, self.t - self.sobol_rng.t)
            self.shifts = rng.integers(0, 2 ** self.t, (self.replications, self.dimension), dtype=int64)
            self.backend_gen = self.mps_sobol_gen
            if not self.scramble:
                warning_s = '''
                Sobol MPS unscrambled samples are not in the domain [0,1)'''
                warnings.warn(warning_s, ParameterWarning)
        elif self.backend == 'pytorch':
            temp_seed = self.seed if self.seed else rng.integers(0, 100, dtype=int64)
            self.sobol_rng = [SobolEngine(dimension=self.dimension, scramble=self.scramble, seed=seed_r)
                                for seed_r in range(temp_seed, temp_seed + self.replications)]
            self.backend_gen = self.pytorch_sobol_gen
            warning_s = '''
                PyTorch SobolEngine issue. See https://github.com/pytorch/pytorch/issues/32047
                    SobolEngine 0^{th} vector is \\vec{.5} rather than \\vec{0}
                    SobolEngine often generates 1 (inclusive) when applying scramble'''
            warnings.warn(warning_s, ParameterWarning)
        else:
            raise ParameterError("Sobol backend must be either 'mps' or 'pytorch'")
        self.mimics = 'StdUniform'
        super().__init__()

    def mps_sobol_gen(self, n_min=0, n_max=8):
        """
        Generate samples from n_min to n_max
        
        Args:
            n_min (int): minimum index. Must be 0 or n_max/2
            n_max (int): maximum index (not inclusive)
        """
        n = int(n_max-n_min)
        x_sob = zeros((n, self.dimension), dtype=int64)
        self.sobol_rng.set_state(n_min)
        for i in range(n):
            next(self.sobol_rng)
            x_sob[i, :] = self.sobol_rng.cur
        if self.scramble:
            x_sob_reps = array([(shift_r ^ (x_sob * 2 ** self.ct)) / 2. ** self.t \
                    for shift_r in self.shifts])
        else:
            x_sob_reps = repeat(x_sob[None, :, :], self.replications, axis=0)
        return x_sob_reps
    
    def pytorch_sobol_gen(self, n_min=0, n_max=8):
        """
        Generate samples from n_min to n_max
        
        Args:
            n_min (int): minimum index. Must be 0 or n_max/2
            n_max (int): maximum index (not inclusive)
        """
        n = int(n_max-n_min)
        for i in range(self.replications):
            self.sobol_rng[i].reset()
            self.sobol_rng[i].fast_forward(n_min)
        return array([self.sobol_rng[i].draw(n).numpy() for i in range(self.replications)])

    def gen_samples(self, n_min=0, n_max=8):
        """
        Generate self.replications (n_max-n_min)xself.d Sobol samples

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
        x_sob_reps = self.backend_gen(n_min,n_max)
        if self.squeeze:
            x_sob_reps = x_sob_reps.squeeze(0)
        return x_sob_reps

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
