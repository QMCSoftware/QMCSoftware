from .._discrete_distribution import DiscreteDistribution
from ...util import ParameterError, ParameterWarning
from numpy import *
from os.path import dirname, abspath
import warnings


class Lattice(DiscreteDistribution):
    """
    Quasi-Random Lattice nets in base 2.

    >>> l = Lattice(2,seed=7)
    >>> l
    Lattice (DiscreteDistribution Object)
        dimension       2^(1)
        randomize       1
        order           natural
        seed            7
        mimics          StdUniform
    >>> l.gen_samples(4)
    array([[0.076, 0.78 ],
           [0.576, 0.28 ],
           [0.326, 0.53 ],
           [0.826, 0.03 ]])
    >>> l.set_dimension(3)
    >>> l.gen_samples(n_min=4,n_max=8)
    array([[0.563, 0.098, 0.353],
           [0.063, 0.598, 0.853],
           [0.813, 0.848, 0.103],
           [0.313, 0.348, 0.603]])
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

        [5] L’Ecuyer, Pierre & Munger, David. (2015).
        LatticeBuilder: A General Software Tool for Constructing Rank-1 Lattice Rules.
        ACM Transactions on Mathematical Software. 42. 10.1145/2754929.
    """

    parameters = ['dimension','randomize','order','seed','mimics']

    def __init__(self, dimension=1, randomize=True, order='natural', seed=None, z_path=None):
        """
        Args:
            dimension (int): dimension of samples
            randomize (bool): If True, apply shift to generated samples. \
                Note: Non-randomized lattice sequence includes the origin.
            order (str): 'linear', 'natural', or 'mps' ordering.
            seed (int): seed the random number generator for reproducibility
            z_path (str): path to generating vector. 
                z_path should be formatted like 'lattice_vec.3600.20.npy' where 'name.d_max.m_max.npy' 
                and d_max is the maximum dimenion and 2^m_max is the max number samples supported
        """
        # set generating matrix
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
        if not z_path:
            self.d_max = 3600
            self.m_max = 20
            self.msb = True
            self.z_full = load(dirname(abspath(__file__))+'/generating_vectors/lattice_vec.3600.20.npy').astype(uint64)
        else:
            if not isfile(z_path):
                raise ParameterError('z_path `' + z_path + '` not found. ')
            self.z_full = load(z_path).astype(uint64)
            f = z_path.split('/')[-1]
            f_lst = f.split('.')
            self.d_max = int(f_lst[1])
            self.m_max = int(f_lst[2])
        self.set_dimension(dimension)
        self.set_seed(seed)
        self.low_discrepancy = True
        self.mimics = 'StdUniform'
        super(Lattice,self).__init__()
    
    def _mps(self, n_min, n_max):
        """ Magic Point Shop Lattice generator. """
        m_low = floor(log2(n_min))+1 if n_min > 0 else 0
        m_high = ceil(log2(n_max))
        gen_block = lambda n: (outer(arange(1, n+1, 2), self.z) % n) / float(n)
        x_lat_full = vstack([gen_block(2**m) for m in range(int(m_low),int(m_high)+1)])
        cut1 = int(floor(n_min-2**(m_low-1))) if n_min>0 else 0
        cut2 = int(cut1+n_max-n_min)
        x = x_lat_full[cut1:cut2,:]
        return x

    def _gail_linear(self, n_min, n_max):
        """ Gail lattice generator in linear order. """
        nelem = n_max - n_min
        if n_min == 0:
            y = arange(0, 1, 1 / nelem).reshape((nelem, 1))
        else:
            y = arange(1 / n_max, 1, 2 / n_max).reshape((nelem, 1))
        x = outer(y, self.z) % 1
        return x

    def _gail_natural(self, n_min, n_max):
        m_low = floor(log2(n_min)) + 1 if n_min > 0 else 0
        m_high = ceil(log2(n_max))
        x_lat_full = vstack([self._gen_block(m) for m in range(int(m_low),int(m_high)+1)])
        cut1 = int(floor(n_min-2**(m_low-1))) if n_min>0 else 0
        cut2 = int(cut1+n_max-n_min)
        x = x_lat_full[cut1:cut2,:]
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
        x = outer(self._vdc(n)+1./(2*n_min),self.z)%1 if n_min>0 else outer(self._vdc(n),self.z)%1
        return x

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True):
        """
        Generate lattice samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples.
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.
            return_non_random (bool): return both the samples with and without randomization

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
        if n_min == 0 and self.randomize==False and warn:
            warnings.warn("Non-randomized lattice sequence includes the origin",ParameterWarning)
        if n_max > 2**self.m_max:
            raise ParameterError('Lattice generating vector supports up to %d samples.'%(2**self.m_max))
        x = self.gen(n_min,n_max)
        if self.randomize:
            x = self.apply_randomization(x)
        return x
    
    def apply_randomization(self, x):
        """
        Apply a digital shift to the samples. 
        
        Args:
            x (ndarray): un-randomized samples to be digitally shifted. 
        
        Return:
            ndarray: x with digital shift aplied.
        """
        x_rand = (x + self.shift)%1
        return x_rand

    def set_seed(self, seed):
        """ See abstract method. """
        self.seed = seed if seed else random.randint(2**32, dtype=uint64)
        random.seed(self.seed)
        self.shift = random.rand(int(self.dimension))
        
    def set_dimension(self, dimension):
        """ See abstract method. """
        self.dimension = dimension
        if self.dimension > self.d_max:
            raise ParameterError('Lattice requires dimension <= %d'%self.d_max)
        self.z = self.z_full[:self.dimension]
        self.shift = random.rand(int(self.dimension))