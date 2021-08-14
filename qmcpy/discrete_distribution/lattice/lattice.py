from .._discrete_distribution import DiscreteDistribution
from ...util import ParameterError, ParameterWarning
from numpy import *
from os.path import dirname, abspath, isfile
import warnings


class Lattice(DiscreteDistribution):
    """
    Quasi-Random Lattice nets in base 2.

    >>> l = Lattice(2,seed=7)
    >>> l.gen_samples(4)
    array([[0.95613942, 0.95613942],
           [0.45613942, 0.45613942],
           [0.20613942, 0.70613942],
           [0.70613942, 0.20613942]])
    >>> l.gen_samples(1)
    array([[0.95613942, 0.95613942]])
    >>> l
    Lattice (DiscreteDistribution Object)
        d               2^(1)
        dvec            [0 1]
        randomize       1
        order           natural
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

    def __init__(self, dimension=1, randomize=True, order='natural', seed=None, 
        generating_vector='lattice_vec.3600.20.npy', d_max=None, m_max=None):
        """
        Args:
           dimension (int or ndarray): dimension of the generator. 
                If an int is passed in, use sequence dimensions [0,...,dimensions-1].
                If a ndarray is passed in, use these dimension indices in the sequence. 
            randomize (bool): If True, apply shift to generated samples. \
                Note: Non-randomized lattice sequence includes the origin.
            order (str): 'linear', 'natural', or 'mps' ordering.
            seed (None or int or numpy.random.SeedSeq): seed the random number generator for reproducibility
            generating_vector (ndarray or str): generating matrix or path to generating matricies. 
                ndarray should have shape (d_max).  
                a string generating_vector should be formatted like 
                'lattice_vec.3600.20.npy' where 'name.d_max.m_max.npy' 
            d_max (int): maximum dimension
            m_max (int): 2^m_max is the max number of supported samples
        
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
            self.z_og = generating_vector
            if d_max is None or m_max is None:
                raise ParameterError("d_max and m_max must be supplied when generating_vector is a ndarray")
            self.d_max = d_max
            self.m_max = m_max
        elif isinstance(generating_vector,str):
            root = dirname(abspath(__file__))+'/generating_vectors/'
            if isfile(root+generating_vector):
                self.z_og = load(root+generating_vector).astype(uint64)
            elif isfile(generating_vector):
                self.z_og = load(generating_vector).astype(uint64)
            else:
                raise ParameterError("generating_vector '%s' not found."%generating_vector)
            parts = generating_vector.split('.')
            self.d_max = int(parts[-3])
            self.m_max = int(parts[-2])
        else:
            raise ParameterError("generating_vector should a ndarray or file path string")
        self.mimics = 'StdUniform'
        self.low_discrepancy = True
        super(Lattice,self).__init__(dimension,seed)
        self.z = self.z_og[self.dvec]
        self.shift = self.rng.uniform(int(self.d))
    
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
        if (n_min!=0 and log2(n_min)%1!=0) or log2(n_max)%1!=0:
            raise Exception("Lattice with linear ordering currently does not support n_min, n_max not powers of 2.")
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
    
    def _spawn(self, s, child_seeds, dimensions):
        return [
            Lattice(
                dimension=dimensions[i],
                randomize=self.randomize,
                order=self.order,
                seed=child_seeds[i],
                generating_vector=self.z_og,
                d_max=self.d_max,
                m_max=self.m_max) 
            for i in range(s)]
