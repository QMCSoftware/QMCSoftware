from .._discrete_distribution import LD
from ...util import ParameterError, ParameterWarning
from ..c_lib import c_lib
import ctypes
from os.path import dirname, abspath, isfile
from numpy import *
import warnings


class DigitalNetB2(LD):
    """
    Quasi-Random digital nets in base 2.
    
    >>> dnb2 = DigitalNetB2(2,seed=7)
    >>> dnb2.gen_samples(4)
    array([[0.56269008, 0.17377997],
           [0.346653  , 0.65070632],
           [0.82074548, 0.95490574],
           [0.10422261, 0.49458097]])
    >>> dnb2.gen_samples(1)
    array([[0.56269008, 0.17377997]])
    >>> dnb2
    DigitalNetB2 (DiscreteDistribution Object)
        d               2^(1)
        dvec            [0 1]
        randomize       LMS_DS
        graycode        0
        entropy         7
        spawn_key       ()
    >>> DigitalNetB2(dimension=2,randomize=False,graycode=True).gen_samples(n_min=2,n_max=4)
    array([[0.75, 0.25],
           [0.25, 0.75]])
    >>> DigitalNetB2(dimension=2,randomize=False,graycode=False).gen_samples(n_min=2,n_max=4)
    array([[0.25, 0.75],
           [0.75, 0.25]])
    >>> dnb2_alpha2 = DigitalNetB2(5,randomize=False,generating_matrices='sobol_mat_alpha2.10600.64.32.lsb.npy')
    >>> dnb2_alpha2.gen_samples(8,warn=False)
    array([[0.      , 0.      , 0.      , 0.      , 0.      ],
           [0.75    , 0.75    , 0.75    , 0.75    , 0.75    ],
           [0.4375  , 0.9375  , 0.1875  , 0.6875  , 0.1875  ],
           [0.6875  , 0.1875  , 0.9375  , 0.4375  , 0.9375  ],
           [0.296875, 0.171875, 0.109375, 0.796875, 0.859375],
           [0.546875, 0.921875, 0.859375, 0.046875, 0.109375],
           [0.234375, 0.859375, 0.171875, 0.484375, 0.921875],
           [0.984375, 0.109375, 0.921875, 0.734375, 0.171875]])

           
    References:

        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.

        [2] Faure, Henri, and Christiane Lemieux. 
        “Implementation of Irreducible Sobol' Sequences in Prime Power Bases.” 
        Mathematics and Computers in Simulation 161 (2019): 13-22. Crossref. Web.

        [3] F.Y. Kuo & D. Nuyens.
        Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients 
        - a survey of analysis and implementation, Foundations of Computational Mathematics, 
        16(6):1631-1696, 2016.
        springer link: https://link.springer.com/article/10.1007/s10208-016-9329-5
        arxiv link: https://arxiv.org/abs/1606.06613
        
        [4] D. Nuyens, `The Magic Point Shop of QMC point generators and generating
        vectors.` MATLAB and Python software, 2018. Available from
        https://people.cs.kuleuven.be/~dirk.nuyens/
        https://bitbucket.org/dnuyens/qmc-generators/src/cb0f2fb10fa9c9f2665e41419097781b611daa1e/cpp/digitalseq_b2g.hpp

        [5] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. 
        (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. 
        In H. Wallach, H. Larochelle, A. Beygelzimer, F. d extquotesingle Alch&#39;e-Buc, E. Fox, & R. Garnett (Eds.), 
        Advances in Neural Information Processing Systems 32 (pp. 8024-8035). Curran Associates, Inc. 
        Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

        [6] I.M. Sobol', V.I. Turchaninov, Yu.L. Levitan, B.V. Shukhman: 
        "Quasi-Random Sequence Generators" Keldysh Institute of Applied Mathematics, 
        Russian Acamdey of Sciences, Moscow (1992).

        [7] Sobol, Ilya & Asotsky, Danil & Kreinin, Alexander & Kucherenko, Sergei. (2011). 
        Construction and Comparison of High-Dimensional Sobol' Generators. Wilmott. 
        2011. 10.1002/wilm.10056. 

        [8] Paul Bratley and Bennett L. Fox. 1988. 
        Algorithm 659: Implementing Sobol's quasirandom sequence generator. 
        ACM Trans. Math. Softw. 14, 1 (March 1988), 88-100. 
        DOI:https://doi.org/10.1145/42288.214372
    """

    dnb2_cf = c_lib.gen_digitalnetb2
    dnb2_cf.argtypes = [
        ctypes.c_ulong,  # n
        ctypes.c_ulong, # n0
        ctypes.c_uint32,  # d
        ctypes.c_uint32, # graycode
        ctypes.c_uint32, # m_max
        ctypes.c_uint32, # t_max
        ctypeslib.ndpointer(ctypes.c_uint64, flags='C_CONTIGUOUS'),  # znew
        ctypes.c_uint32, # set_rshift
        ctypeslib.ndpointer(ctypes.c_uint64, flags='C_CONTIGUOUS'),  # rshift
        ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # x
        ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]  # xr
    dnb2_cf.restype = ctypes.c_uint32

    def __init__(self, dimension=1, randomize='LMS_DS', graycode=False, seed=None, 
        generating_matrices='sobol_mat.21201.32.32.msb.npy', d_max=None, t_max=None, m_max=None, msb=None, t_lms=None, 
        _verbose=False):
        """
        Args:
            dimension (int or ndarray): dimension of the generator. 
                If an int is passed in, use sequence dimensions [0,...,dimensions-1].
                If a ndarray is passed in, use these dimension indices in the sequence. 
            randomize (bool): apply randomization? True defaults to LMS_DS. Can also explicitly pass in
                'LMS_DS': linear matrix scramble with digital shift
                'LMS': linear matrix scramble only
                'DS': digital shift only
            graycode (bool): indicator to use graycode ordering (True) or natural ordering (False)
            seed (list): int seed of list of seeds, one for each dimension.
            generating_matrices (ndarray or str): generating matricies or path to generating matrices.
                ndarray should have shape (d_max, m_max) where each int has t_max bits
                generating_matrices sould be formatted like `gen_mat.21201.32.32.msb.npy` 
                with name.d_max.t_max.m_max.{msb,lsb}.npy
            d_max (int): max dimension
            t_max (int): number of bits in each int of each generating matrix. 
                aka: number of rows in a generating matrix with ints expanded into columns
            m_max (int): 2^m_max is the number of samples supported. 
                aka: number of columns in a generating matrix with ints expanded into columns
            msb (bool): bit storage as ints. e.g. if t_max=3, then 6 is [1 1 0] in MSB (True) and [0 1 1] in LSB (False)
            t_lms (int): LMS scrambling matrix will be t_lms x t_max for generating matrix of shape t_max x m_max
            _verbose (bool): print randomization details
        """
        self.parameters = ['dvec','randomize','graycode']
        if randomize==None or (isinstance(randomize,str) and (randomize.upper()=='NONE' or randomize.upper=='NO')):
            self.set_lms = False
            self.set_rshift = False
        elif isinstance(randomize,bool):
            if randomize:
                self.set_lms = True
                self.set_rshift = True
            else:
                self.set_lms = False
                self.set_rshift = False
        elif randomize.upper() == 'LMS_DS':
            self.set_lms = True
            self.set_rshift = True
        elif randomize.upper() == 'LMS':
            self.set_lms = True
            self.set_rshift = False
        elif randomize.upper() == "DS":
            self.set_lms = False
            self.set_rshift = True
        else:
            msg = '''
                DigitalNetB2' randomize should be either 
                    "LMS_DS" for linear matrix scramble with digital shift or
                    'LMS' for linear matrix scramble only or
                    'DS' for digital shift only. 
            '''
            raise ParameterError(msg)
        self.graycode = graycode
        self.randomize = randomize
        if isinstance(generating_matrices,ndarray):
            self.z_og = generating_matrices
            if d_max is None or t_max is None or m_max is None or msb is None:
                raise ParameterError("d_max, t_max, m_max, and msb must be supplied when generating_matrices is a ndarray")
            self.d_max = d_max
            self.t_max = t_max
            self.m_max = m_max
            self.msb = msb
        elif isinstance(generating_matrices,str):
            root = dirname(abspath(__file__))+'/generating_matrices/'
            if isfile(root+generating_matrices):
                self.z_og = load(root+generating_matrices).astype(uint64)
            elif isfile(generating_matrices):
                self.z_og = load(generating_matrices).astype(uint64)
            else:
                raise ParameterError("generating_matrices '%s' not found."%generating_matrices)
            parts = generating_matrices.split('.')
            self.d_max = int(parts[-5])
            self.t_max = int(parts[-4])
            self.m_max = int(parts[-3])
            self.msb = bool(parts[-2].lower()=='msb')
        else:
            msg = '''
                z_path sould be formatted like `name.d_max.m_max.t_max.msb_or_lsb.npy`
                    d_max is the max dimension, 
                    m_max is such that 2^m_max is the max number of samples supported 
                    t_max is the number of bits in each int of the generating matrix
                    msb_or_lsb is how each int encodes the binary vector
                        e.g. 6 is [1 1 0] in msb and [0 1 1] in lsb
            '''
            raise ParameterError(msg)
        self.t_lms = t_lms if t_lms and self.set_lms else self.t_max
        self._verbose = _verbose
        self.errors = {
            1: 'using natural ordering (graycode=0) where n0 and/or (n0+n) is not 0 or a power of 2 is not allowed.',
            2: 'Exceeding max samples (2^%d) or max dimensions (%d).'%(self.m_max,self.d_max)}
        self.mimics = 'StdUniform'
        self.low_discrepancy = True
        super(DigitalNetB2,self).__init__(dimension,seed)
        if self.t_max>64 or self.t_lms>64 or self.t_max>self.t_lms:
            raise Exception("require t_max <= t_lms <= 64")
        self.z = zeros((self.d,self.m_max),dtype=uint64)
        self.rshift = zeros(self.d,dtype=uint64)
        if not self.msb: # flip bits if using lsb (least significant bit first) order
            for j in range(self.d):
                for m in range(self.m_max):
                    self.z_og[self.dvec[j],m] = self._flip_bits(self.z_og[self.dvec[j],m])
        # set the linear matrix scrambling and random shift
        if self.set_lms and self._verbose: print('s (scrambling_matrix)')
        for j in range(self.d):
            dvecj = self.dvec[j]
            if self.set_lms:
                if self._verbose: print('\n\ts[dvec[%d]]\n\t\t'%j,end='',flush=True)
                for t in range(self.t_lms):
                    t1 = int(minimum(t,self.t_max))
                    u = self.rng.integers(low=0, high=1<<t1, size=1, dtype=uint64)
                    u <<= (self.t_max-t1)
                    if t1<self.t_max: u += 1<<(self.t_max-t1-1)
                    for m in range(self.m_max):
                        v = u&self.z_og[dvecj,m]
                        s = self._count_set_bits(v)%2
                        if s: self.z[j,m] += uint64(1<<(self.t_lms-t-1))
                    if self._verbose:
                        for tprint in range(self.t_max):
                            mask = 1<<(self.t_max-tprint-1)
                            bit = (u&mask)>0
                            print('%-2d'%bit,end='',flush=True)
                        print('\n\t\t',end='',flush=True)
            else:
                self.z[j,:] = self.z_og[dvecj,:]
            if self.set_rshift:
                self.rshift[j] = self.rng.integers(low=0, high=1<<self.t_lms, size=1, dtype=uint64)

    def _flip_bits(self, e):
        """
        flip the int e with self.t_max bits
        """
        u = 0
        for t in range(self.t_max):
            bit = array((1<<t),dtype=uint64)&e
            if bit:
                u += 1<<(self.t_max-t-1)
        return u

    def _count_set_bits(self, e):
        """
        count the number of bits set to 1 in int e
        Brian Kernighan algorithm code: https://www.geeksforgeeks.org/count-set-bits-in-an-integer/
        """
        if (e == 0): return 0
        else: return 1 + self._count_set_bits(e&(e-1)) 

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True, return_unrandomized=False):
        """
        Generate samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples. 
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.
            return_unrandomized (bool): return unrandomized samples as well? 
                If True, return randomized_samples,unrandomized_samples. 
                Note that this only applies when randomize includes Digital Shift.
                Also note that unrandomized samples included linear matrix scrambling if applicable.

        Returns:
            ndarray: (n_max-n_min) x d (dimension) array of samples
        """
        if n:
            n_min = 0
            n_max = n
        if n_min == 0 and self.set_rshift==False and warn:
            warnings.warn("Non-randomized DigitalNetB2 sequence includes the origin",ParameterWarning)
        if return_unrandomized and not self.set_rshift:
            raise ParameterError("return_unrandomized=True only applies when randomize includes a digital shift.")
        n = int(n_max-n_min)
        x = zeros((n,self.d),dtype=double)
        xr = zeros((n,self.d),dtype=double)
        rc = self.dnb2_cf(int(n_min),n,self.d,self.graycode,self.m_max,self.t_lms,self.z,self.set_rshift,self.rshift,x,xr)
        if rc!=0:
            raise ParameterError(self.errors[rc])
        if return_unrandomized:
            return xr,x
        elif self.set_rshift:
            return xr
        else:
            return x
    
    def pdf(self, x):
        """ pdf of a standard uniform """
        return ones(x.shape[0], dtype=float)        
    
    def _spawn(self, child_seed, dimension):
        return DigitalNetB2(
            dimension=dimension,
            randomize=self.randomize,
            graycode=self.graycode,
            seed=child_seed,
            generating_matrices=self.z_og,
            d_max=self.d_max,
            t_max=self.t_max,
            m_max=self.m_max,
            msb=True, # self.z_og is put into MSB during first initialization 
            t_lms=self.t_lms)

class Sobol(DigitalNetB2): pass
