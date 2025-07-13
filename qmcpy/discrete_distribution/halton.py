import warnings
from .abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ..util import ParameterError,ParameterWarning
import qmctoolscl
import numpy as np
from math import *
from copy import deepcopy


class Halton(AbstractLDDiscreteDistribution):
    r"""
    Low discrepancy Halton points.

    Note:
        - The first point of an unrandomized Halton sequence is the origin.
        - QRNG does *not* support multiple replications (independent randomizations).
    
    Examples:
        >>> discrete_distrib = Halton(2,seed=7)
        >>> discrete_distrib(4)
        array([[0.83790457, 0.89981478],
               [0.00986102, 0.4610941 ],
               [0.62236343, 0.02796307],
               [0.29427505, 0.79909098]])
        >>> discrete_distrib
        Halton (AbstractLDDiscreteDistribution)
            d               2^(1)
            replications    1
            randomize       LMS PERM
            t               63
            n_limit         2^(32)
            entropy         7
        
        Replications of independent randomizations 

        >>> x = Halton(3,seed=7,replications=2)(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.70988236, 0.18180876, 0.54073621],
                [0.38178158, 0.61168824, 0.64684354],
                [0.98597752, 0.70650871, 0.31479029],
                [0.15795399, 0.28162992, 0.98945647]],
        <BLANKLINE>
               [[0.620398  , 0.57025403, 0.46336542],
                [0.44021889, 0.69926312, 0.60133428],
                [0.89132308, 0.12030255, 0.35715804],
                [0.04025218, 0.44304244, 0.10724799]]])

        Unrandomized Halton 

        >>> Halton(2,randomize="FALSE",seed=7)(4,warn=False)
        array([[0.        , 0.        ],
               [0.5       , 0.33333333],
               [0.25      , 0.66666667],
               [0.75      , 0.11111111]])
        
        All randomizations 

        >>> Halton(2,randomize="LMS PERM",seed=7)(4)
        array([[0.83790457, 0.89981478],
               [0.00986102, 0.4610941 ],
               [0.62236343, 0.02796307],
               [0.29427505, 0.79909098]])
        >>> Halton(2,randomize="LMS DS",seed=7)(4)
        array([[0.82718745, 0.90603116],
               [0.0303368 , 0.44704107],
               [0.60182684, 0.03580544],
               [0.30505343, 0.78367016]])
        >>> Halton(2,randomize="LMS",seed=7)(4,warn=False)
        array([[0.        , 0.        ],
               [0.82822666, 0.92392942],
               [0.28838899, 0.46493682],
               [0.6165384 , 0.2493814 ]])
        >>> Halton(2,randomize="PERM",seed=7)(4)
        array([[0.11593484, 0.99232505],
               [0.61593484, 0.65899172],
               [0.36593484, 0.32565839],
               [0.86593484, 0.77010283]])
        >>> Halton(2,randomize="DS",seed=7)(4)
        array([[0.56793849, 0.04063513],
               [0.06793849, 0.37396846],
               [0.81793849, 0.7073018 ],
               [0.31793849, 0.15174624]])
        >>> Halton(2,randomize="NUS",seed=7)(4)
        array([[0.141964  , 0.99285569],
               [0.65536579, 0.51938353],
               [0.46955206, 0.11342811],
               [0.78505432, 0.87032345]])
        >>> Halton(2,randomize="QRNG",seed=7)(4)
        array([[0.35362988, 0.38733489],
               [0.85362988, 0.72066823],
               [0.10362988, 0.05400156],
               [0.60362988, 0.498446  ]])
        
        Replications of randomizations 

        >>> Halton(3,randomize="LMS PERM",seed=7,replications=2)(4)
        array([[[0.70988236, 0.18180876, 0.54073621],
                [0.38178158, 0.61168824, 0.64684354],
                [0.98597752, 0.70650871, 0.31479029],
                [0.15795399, 0.28162992, 0.98945647]],
        <BLANKLINE>
               [[0.620398  , 0.57025403, 0.46336542],
                [0.44021889, 0.69926312, 0.60133428],
                [0.89132308, 0.12030255, 0.35715804],
                [0.04025218, 0.44304244, 0.10724799]]])
        >>> Halton(3,randomize="LMS DS",seed=7,replications=2)(4)
        array([[[4.57465163e-01, 5.75419751e-04, 7.47353067e-01],
                [6.29314800e-01, 9.24349881e-01, 8.47915779e-01],
                [2.37544271e-01, 4.63986168e-01, 1.78817056e-01],
                [9.09318567e-01, 2.48566227e-01, 3.17475640e-01]],
        <BLANKLINE>
               [[6.04003127e-01, 9.92849835e-01, 4.21625151e-01],
                [4.57027115e-01, 1.97310094e-01, 2.43670150e-01],
                [8.76467351e-01, 4.22339232e-01, 1.05777101e-01],
                [5.46933622e-02, 7.79075280e-01, 9.29409300e-01]]])
        >>> Halton(3,randomize="LMS",seed=7,replications=2)(4,warn=False)
        array([[[0.        , 0.        , 0.        ],
                [0.82822666, 0.92392942, 0.34057871],
                [0.28838899, 0.46493682, 0.47954399],
                [0.6165384 , 0.2493814 , 0.77045601]],
        <BLANKLINE>
               [[0.        , 0.        , 0.        ],
                [0.93115665, 0.57483093, 0.87170952],
                [0.48046642, 0.8122114 , 0.69381851],
                [0.58055977, 0.28006957, 0.55586147]]])
        >>> Halton(3,randomize="DS",seed=7,replications=2)(4)
        array([[[0.56793849, 0.04063513, 0.74276256],
                [0.06793849, 0.37396846, 0.94276256],
                [0.81793849, 0.7073018 , 0.14276256],
                [0.31793849, 0.15174624, 0.34276256]],
        <BLANKLINE>
               [[0.98309816, 0.80260469, 0.17299622],
                [0.48309816, 0.13593802, 0.37299622],
                [0.73309816, 0.46927136, 0.57299622],
                [0.23309816, 0.9137158 , 0.77299622]]])
        >>> Halton(3,randomize="PERM",seed=7,replications=2)(4)
        array([[[0.11593484, 0.99232505, 0.6010751 ],
                [0.61593484, 0.65899172, 0.0010751 ],
                [0.36593484, 0.32565839, 0.4010751 ],
                [0.86593484, 0.77010283, 0.8010751 ]],
        <BLANKLINE>
               [[0.26543198, 0.12273092, 0.20202896],
                [0.76543198, 0.45606426, 0.60202896],
                [0.01543198, 0.78939759, 0.40202896],
                [0.51543198, 0.23384203, 0.00202896]]])
        >>> Halton(3,randomize="NUS",seed=7,replications=2)(4)
        array([[[0.141964  , 0.99285569, 0.77722918],
                [0.65536579, 0.51938353, 0.22797442],
                [0.46955206, 0.11342811, 0.9975298 ],
                [0.78505432, 0.87032345, 0.57696123]],
        <BLANKLINE>
               [[0.04813634, 0.16158904, 0.56038465],
                [0.89364888, 0.33578478, 0.36145822],
                [0.34111023, 0.84596814, 0.0292313 ],
                [0.71866903, 0.23852281, 0.80431142]]])

    **References:**
    
    1.  Marius Hofert and Christiane Lemieux.  
        qrng: (Randomized) Quasi-Random Number Generators.  
        R package version 0.0-7. (2019).  
        [https://CRAN.R-project.org/package=qrng](https://CRAN.R-project.org/package=qrng).
        
    2.  A. B. Owen.  
        A randomized Halton algorithm in R.  
        [arXiv:1706.02808](https://arxiv.org/abs/1706.02808) [stat.CO]. 2017. 

    3.  A. B. Owen and Z. Pan.  
        Gain coefficients for scrambled Halton points.  
        [arXiv:2308.08035](https://arxiv.org/abs/2308.08035) [stat.CO]. 2023. 
    """

    def __init__(self,
                 dimension = 1,
                 replications = None,
                 seed = None, 
                 randomize = 'LMS PERM',
                 t = 63,
                 n_lim = 2**32,
                 # deprecated
                 t_lms = None):
        r"""
        Args:
            dimension (Union[int,np.ndarray]): Dimension of the generator.

                - If an `int` is passed in, use generating vector components at indices 0,...,`dimension`-1.
                - If an `np.ndarray` is passed in, use generating vector components at these indices.
            
            replications (int): Number of independent randomizations of a pointset.
            seed (Union[None,int,np.random.SeedSeq): Seed the random number generator for reproducibility.
            randomize (str): Options are
                
                - `'LMS PERM'`: Linear matrix scramble with digital shift.
                - `'LMS DS'`: Linear matrix scramble with permutation.
                - `'LMS'`: Linear matrix scramble only.
                - `'PERM'`: Permutation scramble only.
                - `'DS'`: Digital shift only.
                - `'NUS'`: Nested uniform scrambling.
                - `'QRNG'`: Deterministic permutation scramble and random digital shift from QRNG [1] (with `generalize=True`). Does *not* support replications>1.
                - `None`: No randomization. In this case the first point will be the origin. 
            t (int): Number of bits in integer represetation of points *after* randomization. The number of bits in the generating matrices is inferred based on the largest value.
            n_lim (int): Maximum number of compatible points, determines the number of rows in the generating matrices. 
        """
        if t_lms is not None:
            t = t_lms
            warnings.warn("t_lms argument deprecated. Set t instead. Using t = %d"%t,ParameterWarning)
        self.parameters = ['randomize','t','n_limit']
        self.all_primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919],dtype=np.uint64)
        d_limit = len(self.all_primes)
        self.input_t = deepcopy(t) 
        super(Halton,self).__init__(dimension,replications,seed,d_limit,n_lim)
        self.randomize = str(randomize).upper().strip().replace("_"," ")
        if self.randomize=="TRUE": self.randomize = "LMS PERM"
        if self.randomize=="OWEN": self.randomize = "NUS"
        if self.randomize=="NONE": self.randomize = "FALSE"
        if self.randomize=="NO": self.randomize = "FALSE"
        assert self.randomize in ["LMS PERM","LMS DS","LMS","PERM","DS","NUS","QRNG","FALSE"]
        if self.randomize=="QRNG":
            from ._c_lib import _load_c_lib
            assert self.replications==1, "QRNG requires replications=1"
            self.randu_d_32 = self.rng.uniform(size=(self.d,32))
            _c_lib = _load_c_lib()
            import ctypes
            self.halton_cf_qrng = _c_lib.halton_qrng
            self.halton_cf_qrng.argtypes = [
                ctypes.c_int,  # n
                ctypes.c_int,  # d
                ctypes.c_int, # n0
                ctypes.c_int, # generalized
                np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), # res
                np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), # randu_d_32
                np.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')]  # dvec
            self.halton_cf_qrng.restype = None
        self.primes = self.all_primes[self.dvec]
        self.m_max = int(np.ceil(np.log(self.n_limit)/np.log(self.primes.min())))
        self._t_curr = self.m_max
        self.t = self.m_max if self.m_max>t else t
        self.C = qmctoolscl.gdn_get_halton_generating_matrix(np.uint64(1),np.uint64(self.d),np.uint64(self._t_curr))
        if "LMS" in self.randomize:
            S = qmctoolscl.gdn_get_linear_scramble_matrix(self.rng,np.uint64(self.replications),np.uint64(self.d),np.uint64(self._t_curr),np.uint64(self.t),np.uint64(1),self.primes)
            C_lms = np.empty((self.replications,self.d,self.m_max,self.t),dtype=np.uint64)
            qmctoolscl.gdn_linear_matrix_scramble(np.uint64(self.replications),np.uint64(self.d),np.uint64(self.m_max),np.uint64(1),np.uint64(1),np.uint64(self._t_curr),np.uint64(self.t),self.primes,S,self.C,C_lms,backend="c")
            self.C = C_lms
            self._t_curr = self.t
        if "PERM" in self.randomize:
            self.perms = qmctoolscl.gdn_get_digital_permutations(self.rng,np.uint64(self.replications),np.uint64(self.d),self.t,np.uint64(1),self.primes)
        if "DS" in self.randomize:
            self.rshift = qmctoolscl.gdn_get_digital_shifts(self.rng,np.uint64(self.replications),np.uint64(self.d),self.t,np.uint64(1),self.primes)
        if "NUS" in self.randomize:
            new_seeds = self._base_seed.spawn(self.replications*self.d)
            self.rngs = np.array([np.random.Generator(np.random.SFC64(new_seeds[j])) for j in range(self.replications*self.d)]).reshape(self.replications,self.d)
            self.root_nodes = np.array([qmctoolscl.NUSNode_gdn() for i in range(self.replications*self.d)]).reshape(self.replications,self.d)
        assert self.C.ndim==4 and (self.C.shape[0]==1 or self.C.shape[0]==self.replications) and self.C.shape[1]==self.d and self.C.shape[2]==self.m_max and self.C.shape[3]==self._t_curr
        assert 0<self._t_curr<=self.t<=64
        if self.randomize=="FALSE": assert self.C.shape[0]==self.replications, "randomize='FALSE' but replications = %d does not equal the number of sets of generating vectors %d"%(self.replications,self.C.shape[0])

    def _gen_samples(self, n_min, n_max, return_binary, warn):
        if n_min == 0 and self.randomize in ["FALSE","LMS"] and warn:
            warnings.warn("Without randomization, the first Halton point is the origin")
        r_b = np.uint(1)
        r_x = np.uint64(self.C.shape[0])
        n = np.uint64(n_max-n_min)
        d = np.uint64(self.d) 
        n_start = np.uint64(n_min)
        mmax = np.uint64(self.m_max)
        _t_curr = np.uint64(self._t_curr)
        t = np.uint64(self.t)
        bmax = np.uint64(self.primes.max())
        xdig = np.empty((r_x,n,d,_t_curr),dtype=np.uint64)
        qmctoolscl.gdn_gen_natural(r_x,n,d,r_b,mmax,_t_curr,n_start,self.primes,self.C,xdig,backend="c")
        if self.randomize in ["FALSE","LMS"]:
            x = np.empty((r_x,n,d),dtype=np.float64)
            qmctoolscl.gdn_integer_to_float(r_x,n,d,r_b,_t_curr,self.primes,xdig,x,backend="c")
            return x
        if self.randomize=="QRNG": # no replications
            x = np.zeros((self.d,n),dtype=np.double)                        
            self.halton_cf_qrng(n,self.d,int(n_min),True,x,self.randu_d_32,np.int32(self.dvec)) 
            return x.T[None,:,:]
        r = np.uint64(self.replications)
        xdig_new = np.empty((r,n,d,t),dtype=np.uint64)
        if "PERM" in self.randomize:
            qmctoolscl.gdn_digital_permutation(r,n,d,r_x,r_b,_t_curr,t,bmax,self.perms,xdig,xdig_new,backend="c")
        if "DS" in self.randomize:
            qmctoolscl.gdn_digital_shift(r,n,d,r_x,r_b,_t_curr,t,self.primes,self.rshift,xdig,xdig_new,backend="c")
        if "NUS" in self.randomize:
            qmctoolscl.gdn_nested_uniform_scramble(r,n,d,r_x,r_b,_t_curr,t,self.rngs,self.root_nodes,self.primes[None,:],xdig,xdig_new)
        x = np.empty((r,n,d),dtype=np.float64)
        qmctoolscl.gdn_integer_to_float(r,n,d,r_b,t,self.primes,xdig_new,x,backend="c")
        return x         

    def _spawn(self, child_seed, dimension):
        return Halton(
             dimension = dimension,
            replications = None if self.no_replications else self.replications,
            seed = child_seed,
            randomize = self.randomize,
            t = self.input_t,
            n_lim = self.n_limit,
            t_lms = None)
