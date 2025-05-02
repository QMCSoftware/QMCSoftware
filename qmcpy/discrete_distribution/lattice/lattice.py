from ..abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ...util import ParameterError, ParameterWarning
import qmctoolscl
import numpy as np
from numpy.lib.npyio import DataSource
from os.path import dirname, abspath, isfile
import warnings
from copy import deepcopy
from typing import Union


class Lattice(AbstractLDDiscreteDistribution):
    """
    Low discrepancy lattice sequence

    Note:
        - Lattice samples sizes should be powers of 2 e.g. 1, 2, 4, 8, 16, ... 
        - Unrandomized lattices include the origin
    
    Examples:
        >>> l = Lattice(2,seed=7)
        >>> l.gen_samples(4)
        array([[0.04386058, 0.58727432],
               [0.54386058, 0.08727432],
               [0.29386058, 0.33727432],
               [0.79386058, 0.83727432]])
        >>> l.gen_samples(1) # first point in the sequence
        array([[0.04386058, 0.58727432]])
        >>> l
        Lattice (DiscreteDistribution Object)
            d               2^(1)
            dvec            [0 1]
            replications    1
            randomize       SHIFT
            gen_vec         [     1 182667]
            order           NATURAL
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
            replications    1
            randomize       SHIFT
            gen_vec         [       1 11961679]
            order           NATURAL
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
    
    **References**
    
    1.  Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang,
        Lluis Antoni Jimenez Rugama, Da Li, Jagadeeswaran Rathinavel,
        Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou,
        GAIL: Guaranteed Automatic Integration Library (Version 2.3)
        [MATLAB Software], 2019. Available from http://gailgithub.github.io/GAIL_Dev/

    2.  F.Y. Kuo & D. Nuyens.
        Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients
        - a survey of analysis and implementation, Foundations of Computational Mathematics,
        16(6):1631-1696, 2016.
        springer link: https://link.springer.com/article/10.1007/s10208-016-9329-5
        arxiv link: https://arxiv.org/abs/1606.06613

    3.  D. Nuyens, `The Magic Point Shop of QMC point generators and generating
        vectors.` MATLAB and Python software, 2018. Available from
        https://people.cs.kuleuven.be/~dirk.nuyens/

    4.  Constructing embedded lattice rules for multivariate integration
        R Cools, FY Kuo, D Nuyens -  SIAM J. Sci. Comput., 28(6), 2162-2188.

    5.  L'Ecuyer, Pierre & Munger, David. (2015).
        LatticeBuilder: A General Software Tool for Constructing Rank-1 Lattice Rules.
        ACM Transactions on Mathematical Software. 42. 10.1145/2754929.
    """

    def __init__(self,
                 dimension = 1,
                 replications = None,
                 seed = None,
                 randomize = 'SHIFT',
                 generating_vector = "kuo.lattice-33002-1024-1048576.9125.txt",
                 order = 'NATURAL',
                 m_max = None):
        r"""
        Args:
            dimension (Union[int,np.ndarray]): dimension of the generator.

                - If an `int` is passed in, use generating vector components at indices 0,...,`dimension`-1
                - If an `np.ndarray` is passed in, use generating vector components at these indices
            
            replications (int): number of IID randomizations of a pointset
            seed (Union[None,int,np.random.SeedSeq): seed the random number generator for reproducibility
            randomize (bool): If "SHIFT" or True, apply a random shift to the generated samples.
            generating_vector (Union[str,np.ndarray,int]: Specify the generating vector.
                
                - A string `generating_vector` should be the name (or path) of a file from the LDData repo at https://github.com/QMCSoftware/LDData/tree/main/lattice
                - A `np.ndarray` `generating_vector`. Must supply `m_max` where $2^{m_\mathrm{max}}$ is the max number of supported samples. 
                - A positive int `generating_vector`, call it $M$, 
                gives the random generating vector $(1,v_1,\dots,v_{d_\mathrm{max}})^T$ 
                where the $v_i$ are randomly selected from $\{3,5,\dots,2M-1\}$ uniformly and independently. 
                We require require $1 < M < 27$. 

            order (str): "linear", "natural", or "gray" ordering.
            m_max (int): $2^{m_\mathrm{max}}$ is the max number of supported samples
        """
        self.parameters = ['dvec','replications','randomize','gen_vec','order']
        self.input_generating_vector = deepcopy(generating_vector)
        self.input_m_max = deepcopy(m_max)
        # ordering
        self.order = order.upper()
        assert self.order in ['LINEAR','NATURAL','GRAY']
        # generating vector
        if isinstance(generating_vector,str):
            assert generating_vector[-4:]==".txt"
            local_root = dirname(abspath(__file__))+'/generating_vectors/'
            repos = DataSource()
            if repos.exists(local_root+generating_vector):
                datafile = repos.open(local_root+generating_vector)
            elif repos.exists("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/lattice/"+generating_vector):
                datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/lattice/"+generating_vector)
            elif repos.exists("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/"+generating_vector):
                datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/"+generating_vector)
            elif repos.exists("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/lattice/"+generating_vector[7:]):
                datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/lattice/"+generating_vector[7:])
            elif repos.exists("https://raw.githubusercontent.com/QMCSoftware/"+generating_vector):
                datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/"+generating_vector)
            elif repos.exists(generating_vector):
                datafile = repos.open(generating_vector)
            else:
                raise ParameterError("LDData path %s not found"%generating_vector)
            contents = [int(line.rstrip('\n').strip().split("#",1)[0]) for line in datafile.readlines() if line[0]!="#"]
            datafile.close()
            d_limit = contents[0]
            n_limit = contents[1]
            gen_vec = np.array(contents[2:],dtype=np.uint64)[None,:]
        elif isinstance(generating_vector,np.ndarray):
            gen_vec = generating_vector
            if m_max is None:
                raise ParameterError("m_max must be supplied when generating_vector is a np.ndarray")
            n_limit = int(2**m_max)
            d_limit = int(gen_vec.shape[-1])
        elif isinstance(generating_vector,int):
            assert 1<generating_vector<27, "int generating vector out of range"
            n_limit = 2**generating_vector
            assert isinstance(dimension,int), "random generating vector requires int dimension"
            d_limit = dimension
        else:
            raise ParameterError("invalid generating_vector, must be a string, numpy.ndarray, or int")
        super(Lattice,self).__init__(dimension,replications,seed,d_limit,n_limit)
        if isinstance(generating_vector,int):
            m_max = int(np.log2(self.n_limit))
            gen_vec = np.hstack([np.ones((self.replications,1),dtype=np.uint64),2*self.rng.integers(1,2**(m_max-1),size=(self.replications,dimension-1),dtype=np.uint64)+1]).copy()
        assert isinstance(gen_vec,np.ndarray)
        gen_vec = np.atleast_2d(gen_vec) 
        assert gen_vec.ndim==2 and gen_vec.shape[1]>=self.d and (gen_vec.shape[0]==1 or gen_vec.shape[0]==self.replications), "invalid gen_vec.shape = %s"%str(gen_vec.shape)
        # randomization
        self.gen_vec = gen_vec[:,self.dvec].copy()
        self.randomize = str(randomize).upper()
        if self.randomize=="TRUE": self.randomize = "SHIFT"
        if self.randomize=="NONE": self.randomize = "FALSE"
        if self.randomize=="NO": self.randomize = "FALSE"
        assert self.randomize in ["SHIFT","FALSE"]
        if self.randomize=="SHIFT":
            self.shift = self.rng.uniform(size=(self.replications,self.d))

    def _gen_samples(self, n_min, n_max, return_unrandomized, return_binary, warn):
        if return_binary:
            raise ParameterError("Lattice does not support return_binary=True")
        if n_min==0 and self.randomize=="FALSE" and warn:
            warnings.warn("Without randomization, the first lattice point is the origin",ParameterWarning)
        r_x = np.uint64(self.gen_vec.shape[0]) 
        n = np.uint64(n_max-n_min)
        d = np.uint64(self.d) 
        n_start = np.uint64(n_min)
        x = np.empty((r_x,n,d),dtype=np.float64)
        if self.order=="LINEAR":
            assert r_x==1, "lattice linear currently requires there be only 1 generating matrix"
            x = self._gail_linear(n_min,n_max)[None,:,:]
        elif self.order=="NATURAL":
            assert (n_min==0 or (n_min&(n_min-1))==0) and (n_max==0 or (n_max&(n_max-1))==0), "lattice in natural order requires n_min and n_max be 0 or powers of 2"
            qmctoolscl_gen_natural_kwargs = {"backend":"c"}
            _ = qmctoolscl.lat_gen_natural(r_x,n,d,n_start,self.gen_vec,x,**qmctoolscl_gen_natural_kwargs)
        elif self.order=="GRAY":
            qmctoolscl_gen_gray_kwargs = {"backend":"c"}
            _ = qmctoolscl.lat_gen_gray(r_x,n,d,n_start,self.gen_vec,x,**qmctoolscl_gen_gray_kwargs) 
        else: 
            assert False, "invalid lattice order"
        if self.randomize=="FALSE":
            assert return_unrandomized is False, "cannot return_unrandomized when randomize='FALSE'"
            xr = x
        elif self.randomize=="SHIFT":
            r = np.uint64(self.replications)
            qmctoolscl_rand_kwargs = {"backend":"c"}
            xr = np.empty((r,n,d),dtype=np.float64)
            qmctoolscl.lat_shift_mod_1(r,n,d,r_x,x,self.shift,xr,**qmctoolscl_rand_kwargs)
        return (xr,x) if return_unrandomized else xr

    def _gen_block_linear(self, m_next, first=True):
        n = int(2**m_next)
        if first:
            y = np.arange(0, 1, 1 / n).reshape((n, 1))
        else:
            y = np.arange(1 / n, 1, 2 / n).reshape((n, 1))
        x = np.outer(y, self.gen_vec) % 1
        return x

    def calculate_y(self, m_low, m_high, y):
        for m in range(m_low, m_high):
            n = 2 ** m
            y_next = np.arange(1 / n, 1, 2 / n).reshape((int(n / 2), 1))
            temp = np.zeros((n, 1))
            temp[0::2] = y
            temp[1::2] = y_next
            y = temp
        return y

    def _gail_linear(self, n_min, n_max):
        """ Gail lattice generator in linear order. """
        m_low = int(np.floor(np.log2(n_min))) + 1 if n_min > 0 else 0
        m_high = int(np.ceil(np.log2(n_max)))
        if n_min == 0:
            return self._gen_block_linear(m_high, first=True)
        else:
            n = 2 ** (m_low)
            y = np.arange(1 / n, 1, 2 / n).reshape((int(n / 2), 1))
            y = self.calculate_y(m_low, m_high, y)
            x = np.outer(y, self.gen_vec) % 1
            return x

    def _spawn(self, child_seed, dimension):
        return Lattice(
                dimension = dimension,
                replications = self.replications,
                seed = child_seed,
                randomize = self.randomize,
                generating_vector = self.input_generating_vector,
                order = self.order,
                m_max = self.input_m_max,
                )
