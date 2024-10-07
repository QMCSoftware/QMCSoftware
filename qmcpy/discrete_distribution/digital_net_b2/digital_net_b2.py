from .._discrete_distribution import LD
from ...util import ParameterError, ParameterWarning
import qmctoolscl
from os.path import dirname, abspath, isfile
from numpy import *
from numpy.lib.npyio import DataSource
import warnings


class DigitalNetB2(LD):
    """
    Quasi-Random digital nets in base 2.
    
    >>> dnb2 = DigitalNetB2(2,seed=7)
    >>> dnb2.gen_samples(4)
    array([[0.0715562 , 0.07784108],
           [0.81420169, 0.74485558],
           [0.31409299, 0.93233913],
           [0.57163057, 0.26535753]])
    >>> dnb2.gen_samples(1)
    array([[0.0715562 , 0.07784108]])
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
    >>> DigitalNetB2(dimension=3,randomize=False,generating_matrices="LDData/main/dnet/mps.nx_s5_alpha2_m32.txt").gen_samples(8,warn=False)
    array([[0.        , 0.        , 0.        ],
           [0.75841841, 0.45284834, 0.48844557],
           [0.57679828, 0.13226272, 0.10061957],
           [0.31858402, 0.32113875, 0.39369111],
           [0.90278927, 0.45867532, 0.01803333],
           [0.14542431, 0.02548793, 0.4749614 ],
           [0.45587539, 0.33081476, 0.11474426],
           [0.71318879, 0.15377192, 0.37629925]])
    >>> DigitalNetB2(dimension=3,randomize='OWEN',seed=5).gen_samples(8)
    array([[0.33595486, 0.05834975, 0.30066401],
           [0.89110875, 0.84905188, 0.81833285],
           [0.06846074, 0.59997956, 0.67064205],
           [0.6693703 , 0.25824002, 0.10469644],
           [0.44586618, 0.99161977, 0.1873488 ],
           [0.84245267, 0.16445553, 0.56544372],
           [0.18546359, 0.44859876, 0.97389524],
           [0.61215442, 0.64341386, 0.44529863]])

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
        Russian Academy of Sciences, Moscow (1992).

        [7] Sobol, Ilya & Asotsky, Danil & Kreinin, Alexander & Kucherenko, Sergei. (2011). 
        Construction and Comparison of High-Dimensional Sobol' Generators. Wilmott. 
        2011. 10.1002/wilm.10056. 

        [8] Paul Bratley and Bennett L. Fox. 1988. 
        Algorithm 659: Implementing Sobol's quasirandom sequence generator. 
        ACM Trans. Math. Softw. 14, 1 (March 1988), 88-100. 
        DOI:https://doi.org/10.1145/42288.214372
    """

    def __init__(self, dimension=1, randomize='LMS_DS', graycode=False, seed=None, 
        generating_matrices='sobol_mat.21201.32.32.msb.npy', d_max=None, t_max=None, m_max=None, msb=None, t_lms=53, 
        _verbose=False):
        """
        Args:
            dimension (int or :class:`numpy.ndarray`): dimension of the generator. 
                
                - If an int is passed in, use sequence dimensions [0,...,dimension-1].
                - If an ndarry is passed in, use these dimension indices in the sequence. 

            randomize (bool): apply randomization? True defaults to LMS_DS. Can also explicitly pass in
                
                - "LMS_DS": linear matrix scramble with digital shift
                - "LMS": linear matrix scramble only
                - "DS": digital shift only
                - "OWEN" or "NUS": nested uniform scrambling (Owen scrambling)

            graycode (bool): indicator to use graycode ordering (True) or natural ordering (False)
            seed (list): int seed of list of seeds, one for each dimension.
            generating_matrices (:class:`numpy.ndarray` or str): Specify generating matrices. There are a number of optional input types. 
                
                - An ndarray of integers with shape (`d_max`, `m_max)` where each int has `t_max` bits.
                - A string with either a relative path from `LDData repository <https://github.com/QMCSoftware/LDData>`__ (e.g., "LDData/main/dnet/mps.nx_s5_alpha3_m32.txt") or a NumPy file with format "name.d_max.t_max.m_max.{msb,lsb}.npy" (e.g., "gen_mat.21201.32.32.msb.npy")

            d_max (int): max dimension
            t_max (int): number of bits in each int of each generating matrix, aka: number of rows in a generating matrix with ints expanded into columns
            m_max (int): :math:`2^{\\texttt{m\_max}}` is the number of samples supported, aka: number of columns in a generating matrix with ints expanded into columns
            msb (bool): bit storage as ints, e.g., if :math:`{\\texttt{t\_max}} = 3`, then 6 is [1 1 0] in MSB (True) and [0 1 1] in LSB (False)
            t_lms (int): LMS scrambling matrix will be of shape (`t_lms`, `t_max`). Other generating matrix will be of shape (`t_max`, `m_max`).
            _verbose (bool): print randomization details
        """
        self.parameters = ['dvec','randomize','graycode']
        self.randomize = str(randomize).upper()
        if self.randomize=="TRUE": self.randomize = "LMS_DS"
        if self.randomize=="OWEN": self.randomize = "NUS"
        if self.randomize=="NONE": self.randomize = "FALSE"
        if self.randomize=="NO": self.randomize = "FALSE"
        assert self.randomize in ["LMS_DS","LMS","DS","NUS","FALSE"]               
        self.graycode = graycode
        if isinstance(generating_matrices,ndarray):
            self.z_og = generating_matrices
            if d_max is None or t_max is None or m_max is None or msb is None:
                raise ParameterError("d_max, t_max, m_max, and msb must be supplied when generating_matrices is a ndarray")
            self.d_max = d_max
            self.t_max = t_max
            self.m_max = m_max
            self.msb = msb
        elif isinstance(generating_matrices,str):
            parts = generating_matrices.split('.')
            root = dirname(abspath(__file__))+'/generating_matrices/'
            repos = DataSource()
            if isfile(root+generating_matrices):
                self.z_og = load(root+generating_matrices).astype(uint64)
                self.d_max = int(parts[-5])
                self.t_max = int(parts[-4])
                self.m_max = int(parts[-3])
                self.msb = bool(parts[-2].lower()=='msb')
            elif isfile(generating_matrices):
                self.z_og = load(generating_matrices).astype(uint64)
                self.d_max = int(parts[-5])
                self.t_max = int(parts[-4])
                self.m_max = int(parts[-3])
                self.msb = bool(parts[-2].lower()=='msb')
            elif "LDData"==generating_matrices[:6] and repos.exists("https://raw.githubusercontent.com/QMCSoftware/"+generating_matrices):
                datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/"+generating_matrices)
                contents = [line.rstrip('\n').strip() for line in datafile.readlines()]
                self.msb = True
                contents = [line.split("#",1)[0] for line in contents if line[0]!="#"]
                datafile.close()
                assert int(contents[0])==2 # base 2
                self.d_max = int(contents[1])
                self.m_max = int(log2(int(contents[2])))
                self.t_max = int(contents[3])
                compat_shift = max(self.t_max-64,0)
                if compat_shift>0: warnings.warn("Truncating ints in generating matrix to have 64 bits.")
                self.z_og = array([[int(v)>>compat_shift for v in line.split(' ')] for line in contents[4:]],dtype=uint64)
            else:
                raise ParameterError("generating_matrices '%s' not found."%generating_matrices)
        else:
            raise ParameterError("invalid input for generating_matrices, see documentation and doctests.")
        self.t_lms = max(self.t_max,t_lms)
        self._verbose = _verbose
        self.mimics = 'StdUniform'
        self.low_discrepancy = True
        super(DigitalNetB2,self).__init__(dimension,seed)
        if self.t_max>64 or self.t_lms>64 or self.t_max>self.t_lms:
            raise Exception("require t_max <= t_lms <= 64")
        self.z = zeros((self.d,self.m_max),dtype=uint64)
        self.rshift = zeros(self.d,dtype=uint64)
        self.z_og = self.z_og[self.dvec]
        if not self.msb: 
            qmctoolscl.dnb2_gmat_lsb_to_msb(uint64(1),uint64(self.d),uint64(self.m_max),tile(uint64(self.t_max),1),self.z_og,self.z_og)
        if "LMS" in self.randomize:
            self.z = self.z_og.copy()
            S = qmctoolscl.dnb2_get_linear_scramble_matrix(self.rng,uint64(1),uint64(self.d),uint64(self.t_max),uint64(self.t_lms),uint64(self._verbose))
            qmctoolscl.dnb2_linear_matrix_scramble(uint64(1),uint64(self.d),uint64(self.m_max),uint64(1),uint64(self.t_lms),S,self.z_og,self.z)
            self.t_max = self.t_lms
        else:
            self.z = self.z_og
        if "DS" in self.randomize:
            self.rshift = qmctoolscl.random_tbit_uint64s(self.rng,self.t_lms,(self.d,))
        if "NUS" in self.randomize:
            new_seeds = self._base_seed.spawn(self.d)
            self.rngs = array([random.Generator(random.SFC64(new_seeds[j])) for j in range(self.d)]).reshape(1,self.d)
            self.root_nodes = array([qmctoolscl.NUSNode_dnb2() for i in range(self.d)]).reshape(1,self.d)

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True, return_unrandomized=False):
        """
        Generate samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples. 
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.
            return_unrandomized (bool): return unrandomized samples as well? 
                
                - If True, return both randomized samples and unrandomized samples. 
                - If False, return only randomized samples.
                - Note that this only applies when randomize includes Digital Shift.
                - Also note that unrandomized samples included linear matrix scrambling if applicable.

        Returns:
            ndarray: (n_max-n_min) x d (dimension) array of samples
        """
        if n:
            n_min = 0
            n_max = n
        if n_min == 0 and warn and self.randomize in ["FALSE","LMS"]:
            warnings.warn("Non-randomized DigitalNetB2 sequence includes the origin",ParameterWarning)
        if n_max > 2**self.m_max:
            raise ParameterError('DigitalNetB2 generating matrices support up to %d samples.'%(2**self.m_max))
        if return_unrandomized and not self.set_rshift and not self.set_owen:
            raise ParameterError("return_unrandomized=True only applies when randomize includes a digital shift or NUS (Owen) scramble.")
        r = uint64(1) 
        n = uint64(n_max-n_min)
        d = uint64(self.d) 
        n_start = uint64(n_min)
        mmax = uint64(self.m_max)
        xb = empty((n,d),dtype=uint64)
        if self.graycode:
            _ = qmctoolscl.dnb2_gen_gray(r,n,d,n_start,mmax,self.z,xb)
        else: 
            assert (n_min==0 or log2(n_min)%1==0) and (n_max==0 or log2(n_max)%1==0), "DigitalNetB2 in natural order requires n_min and n_max be 0 or powers of 2"
            _ = qmctoolscl.dnb2_gen_natural(r,n,d,n_start,mmax,self.z,xb)
        if return_unrandomized:
            tmaxes_new = tile(uint64(self.t_max),r)
            x = empty((n,d),dtype=float64)
            _ = qmctoolscl.dnb2_integer_to_float(r,n,d,tmaxes_new,xb,x)
        if "DS" in self.randomize:
            r_x = uint64(1)
            lshifts = tile(uint64(self.t_lms-self.t_max),1)
            _ = qmctoolscl.dnb2_digital_shift(r,n,d,r_x,lshifts,xb,self.rshift,xb)
        if "NUS" in self.randomize:
            r_x = 1
            tmax = uint64(self.t_max)
            tmax_new = uint64(self.t_lms)
            xb = xb[None,:,:]
            xrb = zeros((1,n,d),dtype=uint64)
            _ = qmctoolscl.dnb2_nested_uniform_scramble(r,n,d,r_x,tmax,tmax_new,self.rngs,self.root_nodes,xb,xrb)
            xb = xrb[0]
        tmaxes_new = tile(uint64(self.t_lms if self.randomize!="FALSE" else self.t_max),1)
        xr = empty((n,d),dtype=float64) 
        _ = qmctoolscl.dnb2_integer_to_float(r,n,d,tmaxes_new,xb,xr)
        return (xr,x) if return_unrandomized else xr
    
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

