"""
Lattice Generator copied from GAIL (http://gailgithub.github.io/GAIL_Dev/)

Adapted from
    https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/%2Bgail/vdc.m
    https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/%2Bgail/lattice_gen.m

Reference:

    [1] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
    Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou, 
    GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019. 
    Available from http://gailgithub.github.io/GAIL_Dev/

    [2] L’Ecuyer, Pierre & Munger, David. (2015). 
    LatticeBuilder: A General Software Tool for Constructing Rank-1 Lattice Rules. 
    ACM Transactions on Mathematical Software. 42. 10.1145/2754929.
"""

from numpy import *
import os

# generating vector, see Reference [2]
abs_file_path = os.path.join(os.path.dirname(__file__), 'lattice-32001-1024-1048576.3600.npy')
gen_vec = load(abs_file_path).astype(double)
gen_vec_len = len(gen_vec)

def vdc(n):
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

def gen_block(m,z):
    """ Generate samples floor(2**(m-1)) to 2**m. """
    n_min = floor(2**(m-1))
    n = 2**m-n_min
    x = outer(vdc(n)+1./(2*n_min),z)%1 if n_min>0 else outer(vdc(n),z)%1
    return x

def gail_lattice_gen(n_min, n_max, d):
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
        raise Exception('GAIL Lattice has max dimensions %d'%len(gen_vec))
    if n_max > 2**20:
        raise Exception('GAIL Lattice has maximum points 2^20')    
    z = gen_vec[0:d]
    m_low = floor(log2(n_min))+1 if n_min > 0 else 0
    m_high = ceil(log2(n_max))
    x_lat_full = vstack([gen_block(m,z) for m in range(int(m_low),int(m_high)+1)])
    cut1 = int(floor(n_min-2**(m_low-1))) if n_min>0 else 0
    cut2 = int(cut1+n_max-n_min)
    x_lat = x_lat_full[cut1:cut2,:]
    return x_lat