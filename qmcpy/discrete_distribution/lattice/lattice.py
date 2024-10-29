from .._discrete_distribution import LD
from ...util import ParameterError, ParameterWarning
import qmctoolscl
from numpy import *
from os.path import dirname, abspath, isfile
import warnings


class Lattice(LD):
    """
    Quasi-Random Lattice nets in base 2.
    
    >>> l = Lattice(2,seed=7)
    >>> l.gen_samples(4)
    array([[0.04386058, 0.58727432],
           [0.54386058, 0.08727432],
           [0.29386058, 0.33727432],
           [0.79386058, 0.83727432]])
    >>> l.gen_samples(1)
    array([[0.04386058, 0.58727432]])
    >>> l
    Lattice (DiscreteDistribution Object)
        d               2^(1)
        dvec            [0 1]
        randomize       SHIFT
        order           NATURAL
        gen_vec         [     1 182667]
        entropy         7
        spawn_key       ()
    >>> Lattice(dimension=2,randomize=False,order='natural').gen_samples(4, warn=False)
    array([[0.  , 0.  ],
           [0.5 , 0.5 ],
           [0.25, 0.75],
           [0.75, 0.25]])
    >>> Lattice(dimension=2,randomize=False,order='linear').gen_samples(4, warn=False)
    array([[0.  , 0.  ],
           [0.25, 0.75],
           [0.5 , 0.5 ],
           [0.75, 0.25]])
    >>> Lattice(dimension=2,randomize=False,order='gray').gen_samples(4, warn=False)
    array([[0.  , 0.  ],
           [0.5 , 0.5 ],
           [0.75, 0.25],
           [0.25, 0.75]])
    >>> l = Lattice(2,generating_vector=25,seed=55)
    >>> l.gen_samples(4)
    array([[0.84489224, 0.30534549],
           [0.34489224, 0.80534549],
           [0.09489224, 0.05534549],
           [0.59489224, 0.55534549]])
    >>> l
    Lattice (DiscreteDistribution Object)
        d               2^(1)
        dvec            [0 1]
        randomize       SHIFT
        order           NATURAL
        gen_vec         [       1 11961679]
        entropy         55
        spawn_key       ()
    >>> Lattice(dimension=4,randomize=False,seed=353,generating_vector=26).gen_samples(8,warn=False)
    array([[0.   , 0.   , 0.   , 0.   ],
           [0.5  , 0.5  , 0.5  , 0.5  ],
           [0.25 , 0.25 , 0.75 , 0.75 ],
           [0.75 , 0.75 , 0.25 , 0.25 ],
           [0.125, 0.125, 0.875, 0.875],
           [0.625, 0.625, 0.375, 0.375],
           [0.375, 0.375, 0.625, 0.625],
           [0.875, 0.875, 0.125, 0.125]])
    >>> Lattice(dimension=3,randomize=False,generating_vector="LDData/main/lattice/mps.exod2_base2_m20_CKN.txt").gen_samples(8,warn=False)
    array([[0.   , 0.   , 0.   ],
           [0.5  , 0.5  , 0.5  ],
           [0.25 , 0.75 , 0.75 ],
           [0.75 , 0.25 , 0.25 ],
           [0.125, 0.375, 0.375],
           [0.625, 0.875, 0.875],
           [0.375, 0.125, 0.125],
           [0.875, 0.625, 0.625]])
    >>> Lattice(3,seed=7,replications=2).gen_samples(4)
    array([[[0.04386058, 0.58727432, 0.3691824 ],
            [0.54386058, 0.08727432, 0.8691824 ],
            [0.29386058, 0.33727432, 0.1191824 ],
            [0.79386058, 0.83727432, 0.6191824 ]],
    <BLANKLINE>
           [[0.65212985, 0.69669968, 0.10605352],
            [0.15212985, 0.19669968, 0.60605352],
            [0.90212985, 0.44669968, 0.85605352],
            [0.40212985, 0.94669968, 0.35605352]]])
    >>> Lattice(3,seed=7,generating_vector=25,replications=2).gen_samples(4)
    array([[[0.3691824 , 0.65212985, 0.69669968],
            [0.8691824 , 0.15212985, 0.19669968],
            [0.6191824 , 0.90212985, 0.44669968],
            [0.1191824 , 0.40212985, 0.94669968]],
    <BLANKLINE>
           [[0.10605352, 0.63025643, 0.13630282],
            [0.60605352, 0.13025643, 0.63630282],
            [0.35605352, 0.38025643, 0.38630282],
            [0.85605352, 0.88025643, 0.88630282]]])
    
    References:

        [1] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang,
        Lluis Antoni Jimenez Rugama, Da Li, Jagadeeswaran Rathinavel,
        Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou,
        GAIL: Guaranteed Automatic Integration Library (Version 2.3)
        [MATLAB Software], 2019. Available from http://gailgithub.github.io/GAIL_Dev/

        [2] F.Y. Kuo & D. Nuyens.
        Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients
        - a survey of analysis and implementation, Foundations of Computational Mathematics,
        16(6):1631-1696, 2016.
        springer link: https://link.springer.com/article/10.1007/s10208-016-9329-5
        arxiv link: https://arxiv.org/abs/1606.06613

        [3] D. Nuyens, `The Magic Point Shop of QMC point generators and generating
        vectors.` MATLAB and Python software, 2018. Available from
        https://people.cs.kuleuven.be/~dirk.nuyens/

        [4] Constructing embedded lattice rules for multivariate integration
        R Cools, FY Kuo, D Nuyens -  SIAM J. Sci. Comput., 28(6), 2162-2188.

        [5] L'Ecuyer, Pierre & Munger, David. (2015).
        LatticeBuilder: A General Software Tool for Constructing Rank-1 Lattice Rules.
        ACM Transactions on Mathematical Software. 42. 10.1145/2754929.
    """

    def __init__(self, dimension=1, randomize='shift', order='natural', seed=None,
        generating_vector='lattice_vec.3600.20.npy', d_max=None, m_max=None, replications=1):
        """
        Args:
            dimension (int or :class:`numpy.ndarray`): dimension of the generator.

                - If an int is passed in, use sequence dimensions [0,...,dimension-1].
                - If an ndarray is passed in, use these dimension indices in the sequence.
            
            randomize (bool): If True, apply shift to generated samples.
                Note: Non-randomized lattice sequence includes the origin.
            order (str): "linear", "natural", or "gray" ordering.
            seed (None or int or :class:`numpy.random.SeedSeq`): seed the random number generator for reproducibility
            generating_vector (:class:`numpy.ndarray`, str, or int): Specify the generating matrix. There are a number of optional input types. 
                
                - An ndarray of integers.
                - A string `generating_vector` with either a relative path from `LDData repository <https://github.com/QMCSoftware/LDData>`__ (e.g., "LDData/main/lattice/mps.exod2_base2_m20_CKN.txt")  or a NumPy file with format "name.d_max.m_max.npy" (e.g., "lattice_vec.3600.20.npy").
                - An odd integer :math:`1 < M < 27` which creates a random generating vector :math:`[1,v_1,v_2,...,v_{\\texttt{d\_max}}]` where :math:`v_i` is a random integer in :math:`{3,5,...,2*M-1}` supporting up to :math:`2^M` points.
            
            d_max (int): maximum dimension
            m_max (int): :math:`2^{\\texttt{m\_max}}` is the max number of supported samples
            replications (int): number of IID randomizations of a pointset

        Note:
            `d_max` and `m_max` are required if `generating_vector` is an ndarray.
            If `generating_vector` is a string (path), `d_max` and `m_max` can be taken from the file name if None.
        """
        self.parameters = ['dvec','randomize','order','gen_vec']
        self.mimics = 'StdUniform'
        self.low_discrepancy = True
        self.replications_gv = 1
        # ordering
        self.order = order.upper()
        if self.order=="MPS": raise ParameterError("lattice in MPS order is no longer supported")
        assert self.order in ['LINEAR','NATURAL','GRAY']
        # generating vector
        if isinstance(generating_vector,ndarray):
            self.gen_vec_og = generating_vector
            if d_max is None or m_max is None:
                raise ParameterError("d_max and m_max must be supplied when generating_vector is a ndarray")
            self.d_max = d_max
            self.m_max = m_max
            if self.gen_vec_og.ndim==1:
                self.gen_vec_og = self.gen_vec_og[None,:]
            else:
                assert self.gen_vec_og.ndim==2, "ndarray generating vector must be 1 or two dimensional"
                self.replications_gv = len(self.gen_vec_og)
        elif isinstance(generating_vector,int):
            self.m_max = min(26,max(2,generating_vector))
            assert isinstance(dimension,int), "random generating vector requires int dimension"
            self.d_max = dimension
        elif isinstance(generating_vector,str):
            parts = generating_vector.split('.')
            root = dirname(abspath(__file__))+'/generating_vectors/'
            repos = DataSource()
            if isfile(root+generating_vector):
                self.gen_vec_og = load(root+generating_vector).astype(uint64)[None,:]
                self.d_max = int(parts[-3])
                self.m_max = int(parts[-2])
            elif isfile(generating_vector):
                self.gen_vec_og = load(generating_vector).astype(uint64)[None,:]
                self.d_max = int(parts[-3])
                self.m_max = int(parts[-2])
            elif "LDData"==generating_vector[:6]:
                if repos.exists("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/lattice/"+generating_vector[7:]):
                    datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/lattice/"+generating_vector[7:])
                elif repos.exists("https://raw.githubusercontent.com/QMCSoftware/"+generating_vector):
                    datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/"+generating_vector)
                else:
                    raise ParameterError("LDData path %s not found"%generating_vector)
                contents = [int(line.rstrip('\n').strip().split("#",1)[0]) for line in datafile.readlines() if line[0]!="#"]
                datafile.close()
                self.d_max = contents[0]
                n_max = contents[1]; assert log2(n_max)%1==0; self.m_max = int(log2(n_max))
                self.gen_vec_og = array(contents[2:],dtype=uint64)[None,:]
            else:
                raise ParameterError("generating_vector '%s' not found."%generating_vector)
        else:
            raise ParameterError("invalid input for generating_matrices, see documentation and doctests.")
        super(Lattice,self).__init__(dimension,seed)
        # randomization
        if isinstance(generating_vector,int):
            self.replications_gv = replications
            self.gen_vec_og = hstack([ones((replications,1),dtype=uint64),2*self.rng.integers(1,2**(self.m_max-1),size=(replications,dimension-1),dtype=uint64)+1])
        self.gen_vec = self.gen_vec_og[:,self.dvec].copy()
        assert self.gen_vec.ndim==2, "gen_vec must have 2 dimensions"
        self.randomize = str(randomize).upper()
        if self.randomize=="TRUE": self.randomize = "SHIFT"
        if self.randomize=="NONE": self.randomize = "FALSE"
        if self.randomize=="NO": self.randomize = "FALSE"
        assert self.randomize in ["SHIFT","FALSE"]
        if self.randomize=="SHIFT":
            self.shift = self.rng.uniform(size=(replications,self.d))
        if self.replications_gv>1: assert replications==self.replications_gv, "if replications_gv>1 require replications = replications_gv"
        self.replications = replications
        

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True, return_unrandomized=False, qmctoolscl_gen_kwargs={"backend":"c"}, qmctoolscl_rand_kwargs={"backend":"c"}):
        """
        Generate lattice samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples.
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.
            return_unrandomized (bool): return samples without randomization as 2nd return value.
                Will not be returned if randomize=False.
            qmctoolscl_gen_kwargs,qmctoolscl_rand_kwargs (dict): keyword arguments for QMCToolsCL to use OpenCL when generating points or performing randomizations. Defaults to C backend. See https://qmcsoftware.github.io/QMCToolsCL/

        Returns:
            ndarray: replications x (n_max-n_min) x d (dimension) array of samples

        Note:
            Lattice generates in blocks from 2**m to 2**(m+1) so generating
            n_min=3 to n_max=9 requires necessarily produces samples from n_min=2 to n_max=16
            and automatically subsets. May be inefficient for non-powers-of-2 samples sizes.
        """
        if n:
            n_min = 0
            n_max = n            
        if n_min == 0 and self.randomize=="FALSE" and warn:
            warnings.warn("Non-randomized lattice sequence includes the origin",ParameterWarning)
        if n_max > 2**self.m_max:
            raise ParameterError('Lattice generating vector supports up to %d samples.'%(2**self.m_max))
        r_x = uint64(self.replications_gv) 
        n = uint64(n_max-n_min)
        d = uint64(self.d) 
        n_start = uint64(n_min)
        x = empty((r_x,n,d),dtype=float64)
        if self.order=="LINEAR":
            assert r_x==1, "lattice linear currently requires there be only 1 generating matrix"
            x = self._gail_linear(n_min,n_max)[None,:,:]
        elif self.order=="NATURAL":
            assert (n_min==0 or log2(n_min)%1==0) and (n_max==0 or log2(n_max)%1==0), "lattice in natural order requires n_min and n_max be 0 or powers of 2"
            _ = qmctoolscl.lat_gen_natural(r_x,n,d,n_start,self.gen_vec,x,**qmctoolscl_gen_kwargs)
        elif self.order=="GRAY":
            _ = qmctoolscl.lat_gen_gray(r_x,n,d,n_start,self.gen_vec,x,**qmctoolscl_gen_kwargs) 
        else: 
            assert False, "invalid lattice order"
        if r_x==1: x = x[0]
        r = uint64(self.replications)
        if self.randomize=="FALSE":
            assert return_unrandomized is False, "cannot return_unrandomized when randomize='FALSE'"
            return x
        elif self.randomize=="SHIFT":
            xr = empty((r,n,d),dtype=float64)
            qmctoolscl.lat_shift_mod_1(r,n,d,r_x,x,self.shift,xr,**qmctoolscl_rand_kwargs)
            if r==1: xr=xr[0]
            return (xr,x) if return_unrandomized else xr
        else:
            raise ParameterError("incorrect randomize parsing in lattice gen_samples")

    def _gen_block_linear(self, m_next, first=True):
        n = int(2**m_next)
        if first:
            y = arange(0, 1, 1 / n).reshape((n, 1))
        else:
            y = arange(1 / n, 1, 2 / n).reshape((n, 1))
        x = outer(y, self.gen_vec) % 1
        return x

    def calculate_y(self, m_low, m_high, y):
        for m in range(m_low, m_high):
            n = 2 ** m
            y_next = arange(1 / n, 1, 2 / n).reshape((int(n / 2), 1))
            temp = zeros((n, 1))
            temp[0::2] = y
            temp[1::2] = y_next
            y = temp
        return y


    def _gail_linear(self, n_min, n_max):
        """ Gail lattice generator in linear order. """
        m_low = int(floor(log2(n_min))) + 1 if n_min > 0 else 0
        m_high = int(ceil(log2(n_max)))
        if n_min == 0:
            return self._gen_block_linear(m_high, first=True)
        else:
            n = 2 ** (m_low)
            y = arange(1 / n, 1, 2 / n).reshape((int(n / 2), 1))
            y = self.calculate_y(m_low, m_high, y)
            x = outer(y, self.gen_vec) % 1
            return x
        
    def pdf(self, x):
        """ pdf of a standard uniform """
        return ones(x.shape[:-1], dtype=float)

    def _spawn(self, child_seed, dimension):
        return Lattice(
                dimension=dimension,
                randomize=self.randomize,
                order=self.order,
                seed=child_seed,
                generating_vector=self.gen_vec_og,
                d_max=self.d_max,
                m_max=self.m_max,
                replications=self.replications)