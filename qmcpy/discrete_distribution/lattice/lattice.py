from .._discrete_distribution import LD
from ...util import ParameterError, ParameterWarning
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
        randomize       1
        order           natural
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
    >>> Lattice(dimension=2,randomize=False,order='mps').gen_samples(4, warn=False)
    array([[0.  , 0.  ],
           [0.5 , 0.5 ],
           [0.25, 0.75],
           [0.75, 0.25]])
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
        randomize       1
        order           natural
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
    >>> Lattice(dimension=4,randomize=False,seed=353,generating_vector=26,is_parallel=True).gen_samples(8,warn=False)
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

    def __init__(self, dimension=1, randomize=True, order='natural', seed=None,
        generating_vector='lattice_vec.3600.20.npy', d_max=None, m_max=None, is_parallel=True):
        """
        Args:
            dimension (int or ndarray): dimension of the generator.
                If an int is passed in, use sequence dimensions [0,...,dimensions-1].
                If a ndarray is passed in, use these dimension indices in the sequence.
            randomize (bool): If True, apply shift to generated samples.
                Note: Non-randomized lattice sequence includes the origin.
            order (str): 'linear', 'natural', or 'mps' ordering.
            seed (None or int or numpy.random.SeedSeq): seed the random number generator for reproducibility
            generating_vector (ndarray or str or int): Specify the generating matrix. There are a number of optional input types. 
                1) A ndarray of integers. 
                2) A string generating_vector with either: 
                    i)  a relative path from https://github.com/QMCSoftware/LDData e.g. "LDData/main/lattice/mps.exod2_base2_m20_CKN.txt"  or 
                    ii) a numpy file with format "name.d_max.m_max.npy" e.g. 'lattice_vec.3600.20.npy'.
                3) An odd integer 1 < M < 27 which creates a random generating vector [1,v_1,v_2,...,v_{d_max}] where v_i is a random integer in {3,5,...,2*M-1} supporting up to 2^M points.
            d_max (int): maximum dimension
            m_max (int): 2^m_max is the max number of supported samples
            is_parallel (bool): Default to True to perform parallel computations, False serial

        Note:
            d_max and m_max are required if generating_vector is a ndarray.
            If generating_vector is an string (path), d_max and m_max can be taken from the file name if None
        """
        self.parameters = ['dvec','randomize','order']
        self.randomize = randomize
        self.order = order.lower()
        if self.order == 'natural':
            self.gen = self._gail_natural
        elif self.order == 'linear':
            self.gen = self._gail_linear
        elif self.order == 'mps':
            self.gen = self._mps
        else:
            raise Exception("Lattice requires natural, linear, or mps ordering.")
        if isinstance(generating_vector,ndarray):
            self.gen_vec_og = generating_vector
            if d_max is None or m_max is None:
                raise ParameterError("d_max and m_max must be supplied when generating_vector is a ndarray")
            self.d_max = d_max
            self.m_max = m_max
        elif isinstance(generating_vector,int):
            self.m_max = min(26,max(2,generating_vector))
            self.d_max = dimension
        elif isinstance(generating_vector,str):
            parts = generating_vector.split('.')
            root = dirname(abspath(__file__))+'/generating_vectors/'
            repos = DataSource()
            if isfile(root+generating_vector):
                self.gen_vec_og = load(root+generating_vector).astype(uint64)
                self.d_max = int(parts[-3])
                self.m_max = int(parts[-2])
            elif isfile(generating_vector):
                self.gen_vec_og = load(generating_vector).astype(uint64)
                self.d_max = int(parts[-3])
                self.m_max = int(parts[-2])
            elif "LDData"==generating_vector[:6] and repos.exists("https://raw.githubusercontent.com/QMCSoftware/"+generating_vector):
                datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/"+generating_vector)
                contents = [int(line.rstrip('\n').strip().split("#",1)[0]) for line in datafile.readlines() if line[0]!="#"]
                datafile.close()
                self.d_max = contents[0]
                n_max = contents[1]; assert log2(n_max)%1==0; self.m_max = int(log2(n_max))
                self.gen_vec_og = array(contents[2:],dtype=uint64)
            else:
                raise ParameterError("generating_vector '%s' not found."%generating_vector)
        else:
            raise ParameterError("invalid input for generating_matrices, see documentation and doctests.")
        self.mimics = 'StdUniform'
        self.low_discrepancy = True
        super(Lattice,self).__init__(dimension,seed)
        if isinstance(generating_vector,int):
            self.gen_vec_og = append(uint64(1),2*self.rng.integers(1,2**(self.m_max-1),size=dimension-1,dtype=uint64)+1)
        self.gen_vec = self.gen_vec_og[self.dvec]
        self.shift = self.rng.uniform(size=int(self.d))
        self.parameters += ["gen_vec"]
        self.is_parallel = is_parallel


    def _mps(self, n_min, n_max):
        """ Magic Point Shop Lattice generator. """
        m_low = floor(log2(n_min))+1 if n_min > 0 else 0
        m_high = ceil(log2(n_max))

        if not self.is_parallel:
            gen_block = lambda n: (outer(arange(1, n + 1, 2), self.gen_vec) % n) / float(n)
            x_lat_full = vstack([gen_block(2 ** m) for m in range(int(m_low), int(m_high) + 1)])
        else:
            import concurrent.futures
            def gen_point(i, n):
                """ Generate a single lattice point. """
                return ((i * self.gen_vec) % n) / float(n)
            def gen_block_points(m):
                """ Generate a block of points. """
                n = 2 ** m
                return [gen_point(i, n) for i in arange(1, n + 1, 2)]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(gen_block_points, m) for m in range(int(m_low), int(m_high) + 1)]
                # collect the results in the order the futures were created
                x_lat_full = vstack([future.result() for future in futures])

        cut1 = int(floor(n_min - 2 ** (m_low - 1))) if n_min > 0 else 0
        cut2 = int(cut1 + n_max - n_min)
        x = x_lat_full[cut1:cut2, :]
        return x

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

    def _gail_natural(self, n_min, n_max):
        m_low = floor(log2(n_min)) + 1 if n_min > 0 else 0
        m_high = ceil(log2(n_max))
        if not self.is_parallel:
            x_lat_full = vstack([self._gen_block(m) for m in range(int(m_low), int(m_high) + 1)])
        else:
            import concurrent.futures
            # create a ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._gen_block, m) for m in range(int(m_low), int(m_high) + 1)]
            # collect the results in the order the futures (calls) created
            x_lat_full = vstack([future.result() for future in futures])

        cut1 = int(floor(n_min - 2 ** (m_low - 1))) if n_min > 0 else 0
        cut2 = int(cut1 + n_max - n_min)
        x = x_lat_full[cut1:cut2, :]
        return x

    def _vdc(self,n):
        """
        Van der Corput sequence in base 2 where n is a power of 2. We do it this
        way because of our own VDC construction: is much faster and cubLattice
        does not need more.
        """
        k = log2(n)
        q = zeros(int(n))
        for l in range(int(k)):
            nl = 2**l
            kk = 2**(k-l-1)
            ptind_nl = hstack((tile(False,nl),tile(True,nl)))
            ptind = tile(ptind_nl,int(kk))
            q[ptind] += 1./2**(l+1)
        return q

    def _gen_block(self, m):
        """ Generate samples floor(2**(m-1)) to 2**m. """
        n_min = floor(2**(m-1))
        n = 2**m-n_min
        x = outer(self._vdc(n)+1./(2*n_min),self.gen_vec)%1 if n_min>0 else outer(self._vdc(n),self.gen_vec)%1
        return x

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True, return_unrandomized=False):
        """
        Generate lattice samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples.
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.
            return_unrandomized (bool): return samples without randomization as 2nd return value.
                Will not be returned if randomize=False.

        Returns:
            ndarray: (n_max-n_min) x d (dimension) array of samples

        Note:
            Lattice generates in blocks from 2**m to 2**(m+1) so generating
            n_min=3 to n_max=9 requires necessarily produces samples from n_min=2 to n_max=16
            and automatically subsets. May be inefficient for non-powers-of-2 samples sizes.
        """
        if n:
            n_min = 0
            n_max = n
        if return_unrandomized and self.randomize==False:
            raise ParameterError("return_unrandomized=True only applies when when randomize=True.")
        if n_min == 0 and self.randomize==False and warn:
            warnings.warn("Non-randomized lattice sequence includes the origin",ParameterWarning)
        if n_max > 2**self.m_max:
            raise ParameterError('Lattice generating vector supports up to %d samples.'%(2**self.m_max))
        x = self.gen(n_min,n_max)
        if self.randomize:
            xr = (x + self.shift)%1
        if self.randomize==False:
            return x
        elif return_unrandomized:
            return xr, x
        else:
            return xr

    def pdf(self, x):
        """ pdf of a standard uniform """
        return ones(x.shape[0], dtype=float)

    def _spawn(self, child_seed, dimension):
        return Lattice(
                dimension=dimension,
                randomize=self.randomize,
                order=self.order,
                seed=child_seed,
                generating_vector=self.gen_vec_og,
                d_max=self.d_max,
                m_max=self.m_max)