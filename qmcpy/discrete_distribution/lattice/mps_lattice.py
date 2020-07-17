""" 
Lattice sequence generator from Magic Point Shop (https://people.cs.kuleuven.be/~dirk.nuyens/qmc-generators/)

Adapted from https://bitbucket.org/dnuyens/qmc-generators/src/master/python/latticeseq_b2.py

Reference:
    
    [1] F.Y. Kuo & D. Nuyens.
    Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients 
    - a survey of analysis and implementation, Foundations of Computational Mathematics, 
    16(6):1631-1696, 2016.
    springer link: https://link.springer.com/article/10.1007/s10208-016-9329-5
    arxiv link: https://arxiv.org/abs/1606.06613
    
    [2] D. Nuyens, `The Magic Point Shop of QMC point generators and generating
    vectors.` MATLAB and Python software, 2018. Available from
    https://people.cs.kuleuven.be/~dirk.nuyens/

    [3] Constructing embedded lattice rules for multivariate integration
    R Cools, FY Kuo, D Nuyens -  SIAM J. Sci. Comput., 28(6), 2162-2188.

    [4] L’Ecuyer, Pierre & Munger, David. (2015). 
    LatticeBuilder: A General Software Tool for Constructing Rank-1 Lattice Rules. 
    ACM Transactions on Mathematical Software. 42. 10.1145/2754929.
"""

from numpy import *
import os

# generating vector, see Reference [2]
abs_file_path = os.path.join(os.path.dirname(__file__), 'lattice-32001-1024-1048576.3600.npy')
gen_vec = load(abs_file_path).astype(double)
gen_vec_len = len(gen_vec)

def mps_lattice_gen(n_min, n_max, d):
    """
    Generate d dimensionsal lattice samples from n_min to n_max
    
    Args:
        d (int): dimension of the problem, 1<=d<=100.
        n_min (int): minimum index. Must be 0 or n_max/2
        n_max (int): maximum index (not inclusive)
    """
    if n_min==n_max:
        return array([],dtype=double)
    if d > gen_vec_len:
        raise Exception('MPS Lattice has max dimensions %d'%exod2_len)
    if n_max > 2**20:
        raise Exception('MPS Lattice has maximum points 2^20')
    z = gen_vec[:d]
    m_low = floor(log2(n_min))+1 if n_min > 0 else 0
    m_high = ceil(log2(n_max))
    gen_block = lambda n: (outer(arange(1, n+1, 2), z) % n) / float(n)
    x_lat_full = vstack([gen_block(2**m) for m in range(int(m_low),int(m_high)+1)])
    cut1 = int(floor(n_min-2**(m_low-1))) if n_min>0 else 0
    cut2 = int(cut1+n_max-n_min)
    x_lat = x_lat_full[cut1:cut2,:]
    return x_lat
