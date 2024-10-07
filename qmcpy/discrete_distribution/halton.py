import warnings
from ._discrete_distribution import LD
from ..util import ParameterError,ParameterWarning
import qmctoolscl
from numpy import *
from .c_lib import c_lib
import ctypes
from math import *


class Halton(LD):
    """
    Quasi-Random Halton nets.
    
    >>> halton = Halton(2,randomize="LMS_PERM",seed=7)
    >>> halton.gen_samples(4)
    array([[0.2143238 , 0.1243237 ],
           [0.8861583 , 0.5235922 ],
           [0.49631029, 0.84796825],
           [0.66809997, 0.02469234]])
    >>> halton
    Halton (DiscreteDistribution Object)
        d               2^(1)
        dvec            [0 1]
        randomize       LMS_PERM
        entropy         7
        spawn_key       ()
    >>> halton = Halton(2,randomize="LMS_DS",seed=7)
    >>> halton.gen_samples(4)
    array([[0.67057734, 0.75407767],
           [0.49872383, 0.04149125],
           [0.88466123, 0.3669825 ],
           [0.21288306, 0.98844291]])
    >>> halton = Halton(2,randomize="PERM",seed=7)
    >>> halton.gen_samples(4)
    array([[0.11593484, 0.89708776],
           [0.61593484, 0.56375442],
           [0.36593484, 0.23042109],
           [0.86593484, 0.67486554]])
    >>> halton = Halton(2,randomize="DS",seed=7)
    >>> halton.gen_samples(4)
    array([[0.56793849, 0.47926367],
           [0.06793849, 0.81259701],
           [0.81793849, 0.14593034],
           [0.31793849, 0.59037478]])
    >>> halton = Halton(2,randomize="NUS",seed=7)
    >>> halton.gen_samples(4)
    array([[0.141964  , 0.99285569],
           [0.93765172, 0.42142152],
           [0.49784657, 0.11033177],
           [0.55083118, 0.8388036 ]])
    >>> halton = Halton(2,randomize="QRNG",seed=7)
    >>> halton.gen_samples(4)
    array([[0.35362988, 0.38733489],
           [0.85362988, 0.72066823],
           [0.10362988, 0.05400156],
           [0.60362988, 0.498446  ]])
    >>> halton = Halton(2,randomize="FALSE",seed=7)
    >>> halton.gen_samples(4)
    array([[0.        , 0.        ],
           [0.5       , 0.33333333],
           [0.25      , 0.66666667],
           [0.75      , 0.11111111]])

    References:
        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.
        
        [2] Owen, A. B. "A randomized Halton algorithm in R," 2017. arXiv:1706.02808 [stat.CO]

        [3] Owen, A. B., and Pan, Z. "Gain coefficients for scrambled Halton points," 2023. arXiv:2308.08035 [stat.CO]
    """
    halton_cf_qrng = c_lib.halton_qrng
    halton_cf_qrng.argtypes = [
        ctypes.c_int,  # n
        ctypes.c_int,  # d
        ctypes.c_int, # n0
        ctypes.c_int, # generalized
        ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), # res
        ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), # randu_d_32
        ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')]  # dvec
    halton_cf_qrng.restype = None

    def __init__(self, dimension=1, randomize='LMS_PERM', seed=None, t_lms=53, n_lim=2**32):
        """
        Args:
            dimension (int or ndarray): dimension of the generator. 
                If an int is passed in, use sequence dimensions [0,...,dimensions-1].
                If a ndarray is passed in, use these dimension indices in the sequence. 
            randomize (str/bool): select randomization method from 
                
                - "LMS_PERM": linear matrix scramble with digital shift
                - "LMS_DS": linear matrix scramble with permutation
                - "LMS": linear matrix scramble only
                - "PERM": permutation scramble only
                - "DS": digital shift only
                - "OWEN" or "NUS": nested uniform scrambling (Owen scrambling)
                - "QRNG": deterministic permutation scramble and random digital shift from QRNG [1] (with generalize=True) 

            seed (None or int or numpy.random.SeedSeq): seed the random number generator for reproducibility
            n_lim (int): maximum number of compatible points, determines the number of rows in the generating matrices. 
            t_lms (int): number of bits for randomization, defaults to the smallest number of rows required to store 
        """
        self.parameters = ['dvec','randomize']
        self.all_primes = array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919],dtype=uint64)
        self.d_max = len(self.all_primes)
        self.mimics = 'StdUniform'
        self.low_discrepancy = True
        super(Halton,self).__init__(dimension,seed)
        self.randomize = str(randomize).upper()
        if self.randomize=="TRUE": self.randomize = "LMS_PERM"
        if self.randomize=="OWEN": self.randomize = "NUS"
        if self.randomize=="NONE": self.randomize = "FALSE"
        if self.randomize=="NO": self.randomize = "FALSE"
        assert self.randomize in ["LMS_PERM","LMS_DS","LMS","PERM","DS","NUS","QRNG","FALSE"]
        if self.randomize=="QRNG":
            self.randu_d_32 = self.rng.uniform(size=(self.d,32))
        self.primes = self.all_primes[self.dvec]
        self.n_lim = n_lim
        self.m_max = int(ceil(log(n_lim)/log(self.primes.min())))
        self.t_max = self.m_max
        self.t_lms = max(self.m_max,t_lms)
        self.C = qmctoolscl.gdn_get_halton_generating_matrix(uint64(1),uint64(self.d),uint64(self.t_max))
        if "LMS" in self.randomize:
            S = qmctoolscl.gdn_get_linear_scramble_matrix(self.rng,uint64(1),uint64(self.d),uint64(self.t_max),uint64(self.t_lms),uint64(1),self.primes)
            C_lms = empty((self.d,self.m_max,self.t_lms),dtype=uint64)
            qmctoolscl.gdn_linear_matrix_scramble(uint64(1),uint64(self.d),uint64(self.m_max),uint64(1),uint64(1),uint64(self.t_max),uint64(self.t_lms),self.primes,S,self.C,C_lms)
            self.C = C_lms
            self.t_max = self.t_lms
        if "PERM" in self.randomize:
            self.perms = qmctoolscl.gdn_get_permutations(self.rng,uint64(1),uint64(self.d),self.t_lms,uint64(1),self.primes)
        if "DS" in self.randomize:
            self.rshift = qmctoolscl.gdn_get_digital_shifts(self.rng,uint64(1),uint64(self.d),self.t_lms,uint64(1),self.primes)
        if "NUS" in self.randomize:
            new_seeds = self._base_seed.spawn(self.d)
            self.rngs = array([random.Generator(random.SFC64(new_seeds[j])) for j in range(self.d)]).reshape(1,self.d)
            self.root_nodes = array([qmctoolscl.NUSNode_gdn() for i in range(self.d)]).reshape(1,self.d)
             
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
        if n_max >= self.n_lim:
            raise ParameterError("Halton requires n_max <= %d."%self.n_lim)
        if n_min == 0 and self.randomize is False and warn:
            warnings.warn("Unrandomized Halton includes the origin as the first point.")
        if(self.randomize == 'LMS' and warn):
            warnings.warn("LMS Halton includes the origin as the first point.")
        r = uint64(1) 
        r_b = uint(1)
        r_x = uint64(1)
        n = uint64(n_max-n_min)
        d = uint64(self.d) 
        n_start = uint64(n_min)
        mmax = uint64(self.m_max)
        tmax = uint64(self.t_max)
        tmax_new = uint64(self.t_lms)
        bmax = uint64(self.primes.max())
        xdig = empty((n,d,self.t_max),dtype=uint64)
        _ = qmctoolscl.gdn_gen_natural(r,n,d,r_b,mmax,tmax,n_start,self.primes,self.C,xdig)
        if self.randomize=="FALSE":
            x = empty((n,d),dtype=float64)
            _ = qmctoolscl.gdn_integer_to_float(r,n,d,r_b,tmax,self.primes,xdig,x)
            return x
        if self.randomize=="QRNG":
            x = zeros((self.d,n),dtype=double)                        
            self.halton_cf_qrng(n,self.d,int(n_min),True,x,self.randu_d_32,int32(self.dvec)) 
            return x.T  
        xdig_new = empty((n,d,self.t_lms),dtype=uint64)
        if "PERM" in self.randomize:
            _ = qmctoolscl.gdn_digital_permutation(r,n,d,r_x,r_b,tmax,tmax_new,bmax,self.perms,xdig,xdig_new)
        if "DS" in self.randomize:
            _ = qmctoolscl.gdn_digital_shift(r,n,d,r_x,r_b,tmax,tmax_new,self.primes,self.rshift,xdig,xdig_new)
        if "NUS" in self.randomize:
            xdig_new = xdig_new[None,:,:,:]
            _ = qmctoolscl.gdn_nested_uniform_scramble(r,n,d,r_x,r_b,tmax,tmax_new,self.rngs,self.root_nodes,self.primes[None,:],xdig[None,:,:,:],xdig_new)
            xdig_new = xdig_new[0]
        x = empty((n,d),dtype=float64)
        _ = qmctoolscl.gdn_integer_to_float(r,n,d,r_b,tmax_new,self.primes,xdig_new,x)
        return x         

    def pdf(self, x):
        return ones(x.shape[0], dtype=float)

    def _spawn(self, child_seed, dimension):
        return Halton(
            dimension=dimension,
            randomize=self.randomize,
            seed=child_seed,
            t_lms = self.t_lms,
            n_lim = self.n_lim)
