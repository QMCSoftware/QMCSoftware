""" Definition for Sobol, a concrete implementation of DiscreteDistribution """

from .._discrete_distribution import DiscreteDistribution
from .mps_sobol import DigitalSeq
from qrngpy import sobol_qrng
from ...util import ParameterError, ParameterWarning
from numpy import array, int64, log2, repeat, zeros, random
import warnings


class Sobol(DiscreteDistribution):
    """ Quasi-Random Sobol low discrepancy sequence (Base 2) """
    
    parameters = ['dimension','scramble','seed','backend','mimics']

    def __init__(self, dimension=1, scramble=True, seed=None, backend='QRNG'):
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
        if self.backend == 'qrng':
            self.backend_gen = self.qrgn_sobol_gen
        elif self.backend == 'mps':
            self.mps_sobol_rng = DigitalSeq(m=30, s=self.dimension)
            # we guarantee a depth of >=32 bits for shift
            self.t = max(32, self.mps_sobol_rng.t)
            # correction factor to scale the integers
            self.ct = max(0, self.t - self.mps_sobol_rng.t)
            self.backend_gen = self.mps_sobol_gen
            if not self.scramble:
                warning_s = '''
                Sobol MPS unscrambled samples are not in the domain [0,1)'''
                warnings.warn(warning_s, ParameterWarning)
        elif self.backend == 'pytorch':         
            self.backend_gen = self.pytorch_sobol_gen
            warning_s = '''
                PyTorch SobolEngine issue. See https://github.com/pytorch/pytorch/issues/32047
                    SobolEngine 0^{th} vector is \\vec{.5} rather than \\vec{0}
                    SobolEngine sometimes generates 1 after applying scramble'''
            warnings.warn(warning_s, ParameterWarning)
        else:
            raise ParameterError("Sobol backend must be either 'qrng', 'mps', or 'pytorch'")
        self.mimics = 'StdUniform'
        self.reseed(self.seed)
        super().__init__()
        
    def qrgn_sobol_gen(self, n_min=0, n_max=8):
        """
        Generate samples from n_min to n_max
        
        Args:
            n_min (int): minimum index. Must be 0 or n_max/2
            n_max (int): maximum index (not inclusive)
        """
        n = int(n_max-n_min)
        x_sob = sobol_qrng(n,self.dimension,self.scramble,skip=int(n_min),seed=self.seed)
        return x_sob

    def mps_sobol_gen(self, n_min=0, n_max=8):
        """
        Generate samples from n_min to n_max
        
        Args:
            n_min (int): minimum index. Must be 0 or n_max/2
            n_max (int): maximum index (not inclusive)
        """
        n = int(n_max-n_min)
        x_sob = zeros((n, self.dimension), dtype=int64)
        self.mps_sobol_rng.set_state(n_min)
        for i in range(n):
            next(self.mps_sobol_rng)
            x_sob[i, :] = self.mps_sobol_rng.cur
        if self.scramble:
            x_sob = (self.shift ^ (x_sob * 2 ** self.ct)) / 2. ** self.t
        return x_sob
    
    def pytorch_sobol_gen(self, n_min=0, n_max=8):
        """
        Generate samples from n_min to n_max
        
        Args:
            n_min (int): minimum index. Must be 0 or n_max/2
            n_max (int): maximum index (not inclusive)
        """
        from torch.quasirandom import SobolEngine 
        n = int(n_max-n_min)
        se = SobolEngine(dimension=self.dimension, scramble=self.scramble, seed=self.seed)
        se.fast_forward(n_min)
        x_sob = se.draw(n).numpy()
        return x_sob

    def gen_samples(self, n=None, n_min=0, n_max=8):
        """
        Generate (n_max-n_min)xself.d Sobol samples

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
        x_sob = self.backend_gen(n_min,n_max)
        return x_sob

    def reseed(self, new_seed):
        """
        Reseed the generator to get a new scrambling. 
        Args:
            new_seed (int): new seed for generator
        """
        self.seed = new_seed
        if (self.backend == 'qrng' or self.backend == 'pytorch') and (not self.seed):
            # qrng needs a seed, even if it is random
            random.seed(new_seed)
            self.seed = random.randint(0,100000)
        elif self.backend == 'mps':
            random.seed(new_seed)
            self.shift = random.randint(0, 2 ** self.t, self.dimension, dtype=int64)
        

