# coding: utf-8

from .._discrete_distribution import DiscreteDistribution
from .mps_sobol import DigitalSeq
from ..qrng import sobol_qrng
from ...util import ParameterError, ParameterWarning
from numpy import array, int64, log2, repeat, zeros, random
import warnings


class Sobol(DiscreteDistribution):
    """
    Quasi-Random Sobol nets in base 2.
    
    >>> s = Sobol(2,seed=7)
    >>> s
    Sobol (DiscreteDistribution Object)
        dimension       2^(1)
        randomize       1
        seed            7
        backend         qrng
        mimics          StdUniform
        graycode        0
    >>> s.gen_samples(4)
    array([[0.982, 0.883],
           [0.482, 0.383],
           [0.732, 0.133],
           [0.232, 0.633]])
    >>> s.set_dimension(3)
    >>> s.gen_samples(n_min=4,n_max=8)
    array([[0.857, 0.258, 0.226],
           [0.357, 0.758, 0.726],
           [0.607, 0.508, 0.976],
           [0.107, 0.008, 0.476]])
    >>> Sobol(dimension=2,randomize=False,graycode=True).gen_samples(n_min=2,n_max=4)
    array([[0.75, 0.25],
           [0.25, 0.75]])
    >>> Sobol(dimension=2,randomize=False,graycode=False).gen_samples(n_min=2,n_max=4)
    array([[0.25, 0.75],
           [0.75, 0.25]])
           
    References
        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.

        [2] Faure, Henri, and Christiane Lemieux. 
        “Implementation of Irreducible Sobol' Sequences in Prime Power Bases.” 
        Mathematics and Computers in Simulation 161 (2019): 13–22. Crossref. Web.

        [3] F.Y. Kuo & D. Nuyens.
        Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients 
        - a survey of analysis and implementation, Foundations of Computational Mathematics, 
        16(6):1631-1696, 2016.
        springer link: https://link.springer.com/article/10.1007/s10208-016-9329-5
        arxiv link: https://arxiv.org/abs/1606.06613
        
        [4] D. Nuyens, `The Magic Point Shop of QMC point generators and generating
        vectors.` MATLAB and Python software, 2018. Available from
        https://people.cs.kuleuven.be/~dirk.nuyens/
    """
    
    parameters = ['dimension','randomize','seed','backend','mimics','graycode']

    def __init__(self, dimension=1, randomize=True, seed=None, backend='QRNG', graycode=False):
        """
        Args:
            dimension (int): dimension of samples
            randomize (bool): If True, apply digital shift to generated samples.
                Note: Non-randomized Sobol' sequence includes the origin.",ParameterWarning
            seed (int): seed the random number generator for reproducibility
            backend (str): backend generator must be either "QRNG", "MPS", or "PyTorch" 
                "QRNG" is significantly faster, supports optional randomization, and supports optional graycode ordering.
            graycode (bool): indicator to use graycode ordering (True) or natural ordering (False)
        """
        self.dimension = dimension
        self.randomize = randomize
        self.seed = seed
        self.graycode = graycode
        self.backend = backend.lower()
        if self.backend == 'qrng':
            self.backend_gen = self._qrng_sobol_gen
        elif self.backend == 'mps':
            self.mps_sobol_rng = DigitalSeq(m=30, s=self.dimension)
            # we guarantee a depth of >=32 bits for shift
            self.t = max(32, self.mps_sobol_rng.t)
            # correction factor to scale the integers
            self.ct = max(0, self.t - self.mps_sobol_rng.t)
            self.backend_gen = self._mps_sobol_gen
            if not self.randomize:
                warning_s = '''
                Sobol MPS unscrambled samples are not in the domain [0,1)'''
                warnings.warn(warning_s, ParameterWarning)
        elif self.backend == 'pytorch':         
            self.backend_gen = self._pytorch_sobol_gen
            warnings.warn('''
                PyTorch SobolEngine issue. See https://github.com/pytorch/pytorch/issues/32047
                    SobolEngine 0^{th} vector is \\vec{.5} rather than \\vec{0}''',ParameterWarning)
        else:
            raise ParameterError("Sobol backend must be either 'qrng', 'mps', or 'pytorch'")
        if self.backend != 'qrng' and (not self.graycode):
            warning_s = '''
                %s backend does not (yet) support natural ordering.
                Using graycode ordering. Use "QRNG" backend for natural ordering'''%self.backend
            warnings.warn(warning_s,ParameterWarning)
        self.mimics = 'StdUniform'
        self.set_seed(self.seed)
        super(Sobol,self).__init__()
        
    def _qrng_sobol_gen(self, n_min=0, n_max=8):
        """
        Generate samples from n_min to n_max
        
        Args:
            n_min (int): minimum index. Must be 0 or n_max/2
            n_max (int): maximum index (not inclusive)
        """
        n = int(n_max-n_min)
        x_sob = sobol_qrng(n,self.dimension,self.randomize,skip=int(n_min),graycode=self.graycode,seed=self.seed)
        return x_sob

    def _mps_sobol_gen(self, n_min=0, n_max=8):
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
        if self.randomize:
            x_sob = (self.shift ^ (x_sob * 2 ** self.ct)) / 2. ** self.t
        return x_sob
    
    def _pytorch_sobol_gen(self, n_min=0, n_max=8):
        """
        Generate samples from n_min to n_max
        
        Args:
            n_min (int): minimum index.
            n_max (int): maximum index. 
        """
        import torch
        from torch.quasirandom import SobolEngine
        n = int(n_max-n_min)
        se = SobolEngine(dimension=self.dimension, scramble=self.randomize, seed=self.seed)
        se.fast_forward(n_min)
        x_sob = se.draw(n,dtype=torch.float64).numpy()
        return x_sob

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True):
        """
        Generate samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples. 
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.

        Returns:
            ndarray: (n_max-n_min) x d (dimension) array of samples
        """
        if n:
            n_min = 0
            n_max = n
        if n_min == 0 and self.backend in ['qrng','mps'] and self.randomize==False and warn:
            warnings.warn("Non-randomized Sobol' sequence includes the origin",ParameterWarning)
        x_sob = self.backend_gen(n_min,n_max)
        return x_sob

    def set_seed(self, seed):
        """ See abstract method. """
        self.seed = seed
        if (self.backend == 'qrng' or self.backend == 'pytorch') and (not self.seed):
            # qrng needs a seed, even if it is random
            random.seed(seed)
            self.seed = random.randint(2**32)
        elif self.backend == 'mps':
            random.seed(seed)
            self.shift = random.randint(0, 2 ** self.t, self.dimension, dtype=int64)
        
    def set_dimension(self, dimension):
        """
        See abstract method. 

        Note:
            Computes a new random shift to be applied to samples
        """
        self.dimension = dimension
        if self.backend == 'mps':
            self.mps_sobol_rng = DigitalSeq(m=30, s=self.dimension)
            # we guarantee a depth of >=32 bits for shift
            self.t = max(32, self.mps_sobol_rng.t)
            # correction factor to scale the integers
            self.ct = max(0, self.t - self.mps_sobol_rng.t)
            self.backend_gen = self._mps_sobol_gen
            self.shift = random.randint(0, 2 ** self.t, self.dimension, dtype=int64)

