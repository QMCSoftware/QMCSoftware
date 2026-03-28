import qmctoolscl 
import math
import numpy as np
from scipy.stats import kurtosis,norm
import pandas as pd
import multiprocess
import os
from mpi4py import MPI

""" Faure code """
class Faure:
    def __init__(self, d, p, m, seed=11):
        """
        Initialize Faure sequence generator
        
        Parameters:
        -----------
        d : int
            Dimension of the sequence
        p: int
            smallest prime >=d. the base used to generate the points
        m : int
            Exponent for number of points (n = p^m)
        seed : int
            Random seed for reproducibility
        """
        self.d = d
        self.m = m
        if hasattr(seed, 'generate_state'):
            self.seed = int(seed.generate_state(1)[0])
        else:
            self.seed = int(seed)
        
        # Find prime base
        self.p = p
        self.n = int(self.p ** m)
        
        # Calculate number of digits needed 
        eps = np.finfo(np.float64).tiny
        self.t = int(np.ceil(np.emath.logn(p,p-1)-np.emath.logn(p,eps)))
        self.t = max(m, min(self.t, 64))
        
        # Construct generating matrices (used by both NUS and LMS)
        self._construct_generating_matrices()
        
        self.bases = np.full(d, self.p, dtype=np.uint64)
    
    
    def _construct_generating_matrices(self):
        """Construct base generating matrices"""
        self.C = np.ones((self.d,1,1),dtype=np.uint64)*np.eye(self.m,dtype=np.uint64)
        
        if self.d > 1:
            for a in range(self.m):
                for b in range(a+1):
                    self.C[1, a, b] = math.comb(a, b) % self.p
        
        if self.d > 2:
            for k in range(2, self.d):
                for a in range(self.m):
                    for b in range(a+1):
                        self.C[k, a, b] = (int(self.C[1,a,b])*((k**(a-b))%self.p))%self.p
    
    def gen_samples(self, r, method="LMS"):
        """
        Generate randomized Faure samples
        
        Parameters:
        -----------
        r : int
            Number of independent randomizations to generate
        method : str, default="LMS"
            Randomization method: "NUS" or "LMS"
            
        Returns:
        --------
        samples : ndarray of shape (r, n, d)
            Array of randomized Faure points in [0,1]^d
        """
        if method == "NUS":
            xrdig = self._apply_nus(r)
        elif method == "LMS":
            xrdig = self._apply_lms(r)
        else:
            raise ValueError(f"method must be 'NUS' or 'LMS', got '{method}'")
        
        # Convert digits to floats
        samples = np.zeros((r, self.n, self.d), dtype=np.float64)
        qmctoolscl.gdn_integer_to_float(
            np.uint64(r), np.uint64(self.n), np.uint64(self.d), np.uint64(1), 
            np.uint64(self.t), self.bases, xrdig, samples
        )
        
        return samples
    
    def _apply_nus(self, r):
        """Apply Nested Uniform Scrambling"""
        # Generate unrandomized points
        xdig = np.zeros((self.n, self.d, self.m), dtype=np.uint64)
        qmctoolscl.gdn_gen_natural(
            np.uint64(1), np.uint64(self.n), np.uint64(self.d), np.uint64(1),
            np.uint64(self.m), np.uint64(self.m), np.uint64(0), 
            self.bases, self.C, xdig
        )
        
        # Apply NUS
        base_seed_seq = np.random.SeedSequence(self.seed)
        seeds = base_seed_seq.spawn(r * self.d)
        rngs = np.array([np.random.Generator(np.random.SFC64(s)) for s in seeds]).reshape(r,self.d)
        root_nodes = np.array([qmctoolscl.NUSNode_gdn() for i in range(r*self.d)]).reshape(r,self.d)
        xrdig = np.zeros((r, self.n, self.d, self.t), dtype=np.uint64)
        qmctoolscl.gdn_nested_uniform_scramble(
            np.uint64(r), np.uint64(self.n), np.uint64(self.d), 
            np.uint64(1), np.uint64(1), np.uint64(self.m), np.uint64(self.t), 
            rngs, root_nodes, self.bases[None], xdig[None], xrdig
        )
        
        return xrdig
    
    def _apply_lms(self, r):
        """Apply Linear Matrix Scrambling"""
        rng = np.random.Generator(np.random.SFC64(self.seed))
        
        # Get scrambling matrix
        S = qmctoolscl.gdn_get_linear_scramble_matrix(
            rng, np.uint64(r), np.uint64(self.d), np.uint64(self.m), 
            np.uint64(self.t), np.uint64(1), self.bases[None]
        )
        
        # Apply LMS to C
        C_lms = np.zeros((r, self.d, self.m, self.t), dtype=np.uint64)
        qmctoolscl.gdn_linear_matrix_scramble(
            np.uint64(r), np.uint64(self.d), np.uint64(self.m), 
            np.uint64(1), np.uint64(1), np.uint64(self.m), np.uint64(self.t), 
            self.bases, S, self.C, C_lms
        )
        
        # Generate points with scrambled C
        xdig = np.zeros((r, self.n, self.d, self.t), dtype=np.uint64)
        qmctoolscl.gdn_gen_natural(
            np.uint64(r), np.uint64(self.n), np.uint64(self.d), np.uint64(1),
            np.uint64(self.m), np.uint64(self.t), np.uint64(0), 
            self.bases, C_lms, xdig
        )
        
        # Apply digital permutations
        perms = qmctoolscl.gdn_get_digital_permutations(
            rng, np.uint64(r), np.uint64(self.d), np.uint64(self.t), 
            np.uint64(1), self.bases
        )
        
        xrdig = np.zeros((r, self.n, self.d, self.t), dtype=np.uint64)
        qmctoolscl.gdn_digital_permutation(
            np.uint64(r), np.uint64(self.n), np.uint64(self.d), 
            np.uint64(r), np.uint64(1), np.uint64(self.t), np.uint64(self.t), 
            np.uint64(self.p), perms, xdig, xrdig
        )
        
        return xrdig

""" Kurtosis Calculation Function"""
def kurt_calc(unique_p,p,d,m,gs,child_seed, m_repeat, R_fixed, method = "NUS"):
    print(f"unique_p_val={unique_p}, m_repeat={m_repeat}\n")
    max_index = 0
    for j in range (len(d)):
        if (p[j] == unique_p) and (d[j] > d[max_index]):
            max_index = j
    samples = Faure(d[max_index],unique_p,(m - 1),seed=child_seed).gen_samples(R_fixed,method=method)
    x_qmc_norm = norm.ppf(samples)
    len_d = np.sum(p == unique_p)
    d_ind = np.where(p == unique_p) [0]
    kurt_arr = np.empty((len(gs),len_d,m))
    for i in range(len_d):
        for n in range (m):
            w_qmc = x_qmc_norm[:, :int(unique_p**n), :d[d_ind[i]]].sum(axis = 2) / np.sqrt(d[d_ind[i]])
            counter = 0
            for g in gs.values():
                y = g(w_qmc).mean(axis = 1)
                kurt_arr[counter, i, n] = kurtosis(y, bias=False,fisher=False)
                counter = counter + 1
    np.save(f"tmp_{int(unique_p)}_{m_repeat}.npy", kurt_arr)

""" The ridge functions """
def g_jmp(w):
    return w >= 1

def g_knk(w):
    return ((np.minimum(np.maximum(-2, w), 1)) + 2) / 3

def g_smo(w):
    return norm.cdf(w + 1)

def g_fin(w):
    return np.minimum(1, ((np.sqrt(np.maximum(w + 2, 0))) / 2))

""" Running the experiments """

if __name__ == "__main__":    
    """ The parameters for the experiment """
    alpha = 0.05 # Significance level, confidence level = 1 - alpha

    # The ridge functions:
    gs = {
    "jmp": g_jmp,
    "knk": g_knk,
    "smo": g_smo,
    "fin": g_fin,}

    d = np.array([1,2,4,6]) # The different d's to test on

    #kappa = 6 # kurtosis bound

    #R_1_min = (kappa - 1)/alpha # A lower bound for R_1

    #R_vary = (R_1_min * np.arange(1, 11)).astype(int) # The different number of replications to test

    R_fixed = 1000 # The fixed R to be used for the kurtosis experiments

    M = 100  # The number of times the experiments will be independently repeated

    m = 4 # m different n's will be tested where n is the number of RQMC samples per replication

    """ Primes code """

    ALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919]
    def get_next_prime(d):
        """Find the smallest prime >= d"""
        for p in ALL_PRIMES:
            if p >= d:
                return p
        raise ValueError(f"No prime found >= {d} in precomputed list")

    # getting the bases p based on dimension

    p = np.empty((len(d)))
    for j in range(len(d)):
        p[j] = get_next_prime(d[j])
    unique_p = np.unique(p)

    # seed settings

    global_seed = 7
    parent_seed = np.random.SeedSequence(global_seed)
    all_seeds = np.empty((len(unique_p), M), dtype=object)
    for i in range(len(unique_p)):
        for m_repeat in range(M):
            all_seeds[i, m_repeat] = parent_seed.spawn(1)[0]
    
    """ Using the ridge functions """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    args = [
    (unique_p[i], p, d, m, gs, all_seeds[i, m_repeat], m_repeat, R_fixed)
    for i in range(len(unique_p))
    for m_repeat in range(M)
    if unique_p[i] == 7
    and m_repeat == 3]
    # split args across MPI nodes
    my_args = [args[i] for i in range(len(args)) if i % size == rank]
    print(f"Node {rank}/{size} running {len(my_args)} jobs")
    with multiprocess.Pool() as pool:
        pool.starmap(kurt_calc, my_args)