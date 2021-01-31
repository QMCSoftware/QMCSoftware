from .._discrete_distribution import DiscreteDistribution
from ...util import ParameterError, ParameterWarning
from ..c_lib import c_lib
import ctypes
from os.path import dirname, abspath, isfile
from numpy import *
import warnings


class Sobol(DiscreteDistribution):
    """
    Quasi-Random Sobol nets in base 2.
    
    >>> s = Sobol(2,seed=7)
    >>> s.gen_samples(4)
    array([[0.809, 0.278],
           [0.308, 0.566],
           [0.661, 0.862],
           [0.159, 0.044]])
    >>> s.gen_samples(1)
    array([[0.809, 0.278]])
    >>> s
    Sobol (DiscreteDistribution Object)
        d               2^(1)
        randomize       1
        graycode        0
        seed            585903
        mimics          StdUniform
        dim0            0
    >>> Sobol(dimension=2,randomize=False,graycode=True).gen_samples(n_min=2,n_max=4)
    array([[0.75, 0.25],
           [0.25, 0.75]])
    >>> Sobol(dimension=2,randomize=False,graycode=False).gen_samples(n_min=2,n_max=4)
    array([[0.25, 0.75],
           [0.75, 0.25]])
           
    References:

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

        [5] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. 
        (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. 
        In H. Wallach, H. Larochelle, A. Beygelzimer, F. d extquotesingle Alch&#39;e-Buc, E. Fox, & R. Garnett (Eds.), 
        Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. 
        Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

        [6] I.M. Sobol', V.I. Turchaninov, Yu.L. Levitan, B.V. Shukhman: 
        "Quasi-Random Sequence Generators" Keldysh Institute of Applied Mathematics, 
        Russian Acamdey of Sciences, Moscow (1992).

        [7] Sobol, Ilya & Asotsky, Danil & Kreinin, Alexander & Kucherenko, Sergei. (2011). 
        Construction and Comparison of High-Dimensional Sobol' Generators. Wilmott. 
        2011. 10.1002/wilm.10056. 

        [8] Paul Bratley and Bennett L. Fox. 1988. 
        Algorithm 659: Implementing Sobol's quasirandom sequence generator. 
        ACM Trans. Math. Softw. 14, 1 (March 1988), 88–100. 
        DOI:https://doi.org/10.1145/42288.214372
    """

    def __init__(self, dimension=1, randomize='LMS', graycode=False, seed=None, z_path=None, dim0=0):
        """
        Args:
            dimension (int): dimension of samples
            randomize (bool): Apply randomization? True defaults to LMS. Can also explicitly pass in
                'LMS': Linear matrix scramble with DS 
                'DS': Just Digital Shift
            graycode (bool): indicator to use graycode ordering (True) or natural ordering (False)
            seeds (list): int seed of list of seeds, one for each dimension.
            z_path (str): path to generating matricies. 
                z_path sould be formatted like `gen_mat.21201.32.msb.npy` with name.d_max.m_max.msb_or_lsb.npy
            dim0 (int): first dimension
        """
        self.parameters = ['d','randomize','graycode','seed','mimics','dim0']
        # initialize c code
        self.get_unsigned_long_long_size_cf = c_lib.get_unsigned_long_long_size
        self.get_unsigned_long_long_size_cf.argtypes = []
        self.get_unsigned_long_long_size_cf.restype = ctypes.c_uint8
        self.get_unsigned_long_size_cf = c_lib.get_unsigned_long_size
        self.get_unsigned_long_size_cf.argtypes = []
        self.get_unsigned_long_size_cf.restype = ctypes.c_uint8
        self.sobol_cf = c_lib.sobol
        self.sobol_cf.argtypes = [
            ctypes.c_ulong,  # n
            ctypes.c_uint32,  # d
            ctypes.c_ulong, # n0
            ctypes.c_uint32, # d0
            ctypes.c_uint32,  # randomize
            ctypes.c_uint32, # graycode
            ctypeslib.ndpointer(ctypes.c_uint64, flags='C_CONTIGUOUS'), # seeds
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # x (result)
            ctypes.c_uint32, # d_max
            ctypes.c_uint32, # m_max
            ctypeslib.ndpointer(ctypes.c_uint64, flags='C_CONTIGUOUS'),  # z (generating matrix)
            ctypes.c_uint32, # msb
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # xjlms (result)
            ctypes.c_uint32] # set_xjlms
        # set parameters
        self.sobol_cf.restype = ctypes.c_uint32
        self._set_dimension(dimension)
        self.set_seed(seed)
        self.set_randomize(randomize)
        self.set_graycode(graycode)
        self.set_dim0(dim0)
        # set generating matrix
        if not z_path:
            self.d_max = 21201
            self.m_max = 32
            self.msb = True
            self.z = load(dirname(abspath(__file__))+'/generating_matricies/sobol_mat.21201.32.msb.npy').astype(uint64)
        else:
            if not isfile(z_path):
                raise ParameterError('z_path `' + z_path + '` not found. ')
            self.z = load(z_path).astype(uint64)
            f = z_path.split('/')[-1]
            f_lst = f.split('.')
            self.d_max = int(f_lst[1])
            self.m_max = int(f_lst[2])
            msblsb = f_lst[3].lower()
            if msblsb == 'msb':
                self.msb = True
            elif msblsb == 'lsb':
                self.msb = False
            else:
                msg = '''
                    z_path sould be formatted like `sobol_mat.21201.32.msb.npy`
                    with name.d_max.m_max.msb_or_lsb.npy
                '''
                raise ParameterError(msg)
        self.errors = {
            1: 'requires 32 bit precision but system has unsigned int with < 32 bit precision.',
            2: 'using natural ordering (graycode=0) where n0 and/or (n0+n) is not 0 or a power of 2 is not allowed.',
            3: 'Exceeding max samples (2^%d) or max dimensions (%d).'%(self.m_max,self.d_max)}
        self.low_discrepancy = True
        self.mimics = 'StdUniform'
        super(Sobol,self).__init__()        

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True, return_jlms=False):
        """
        Generate samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples. 
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.
            return_jlms (bool): return the LMS matrix without digital shift. 
                Only applies when randomize='LMS' (the default). 

        Returns:
            ndarray: (n_max-n_min) x d (dimension) array of samples
        """
        if n:
            n_min = 0
            n_max = n
        if n_min == 0 and self.randomize==False and warn:
            warnings.warn("Non-randomized AGS Sobol sequence includes the origin",ParameterWarning)
        if return_jlms and self.randomize!=1:
            raise ParameterError("return_jlms=True only applies when randomize='LMS'.")
        n = int(n_max-n_min)
        x = zeros((n,self.d), dtype=double)
        xjlms = zeros((n,self.d), dtype=double)
        seeds = random.RandomState(self.seed).choice(1000000, self.d, replace=False).astype(uint64)+1
        rc = self.sobol_cf(n, self.d, int(n_min), self.dim0, self.randomize, self.graycode, \
            seeds, x, self.d_max, self.m_max, self.z, self.msb, xjlms, return_jlms)
        if rc!= 0:
            raise ParameterError(self.errors[rc])
        if self.randomize==1 and return_jlms:
            return x,xjlms
        else:
            return x
    
    def pdf(self, x):
        """ pdf of a standard uniform """
        return ones(x.shape[0], dtype=float)

    def set_seed(self, seed=None):
        """
        Reset the seeds

        Args:
            seeds (int): new seed
        """
        self.rng = random.RandomState(seed)
        self.seed = self.rng.choice(1000000,1)
            
    def _set_dimension(self, dimension):
        """
        Reset the dimension

        Args:
            dimension (int): new dimension
        """
        self.d = dimension

    def set_randomize(self, randomize):
        """
        Reset the randomization

        Args:
            randomize (str): randomization type. Either 
                'LMS': linear matrix scramble with digital shift
                'DS': just the digital shift
        """
        if randomize==None or (isinstance(randomize,str) and (randomize.upper()=='NONE' or randomize.upper=='No')):
            self.randomize = 0
        elif isinstance(randomize,bool):
            self.randomize = int(randomize)
        elif randomize.upper() in ["LMS","LINEAR MATRIX SCRAMBLE"]:
            self.randomize = 1
        elif randomize.upper() in ["DS","DIGITAL SHIFT"]:
            self.randomize = 2
        else:
            msg = '''
                Sobol' randomize should be either 
                    'LMS' for Linear Matrix Scramble or 
                    'DS' for Digital Shift. 
            '''
            raise ParameterError(msg)
    
    def set_graycode(self, graycode):
        """
        Reset the graycode

        Args:
            graycode (bool): use graycode?
        """
        self.graycode = graycode
    
    def set_dim0(self, dim0):
        """
        Reset the first dimension

        Args:
            dim0 (int): first dimension
        """
        self.dim0 = dim0

DigitalNet = Sobol
