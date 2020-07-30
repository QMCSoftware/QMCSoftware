"""
Lattice Generator from GAIL (http://gailgithub.github.io/GAIL_Dev/)

Original Implementations:

    [a] https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/%2Bgail/vdc.m
    
    [b] https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/%2Bgail/lattice_gen.m

References:

    [1] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
    Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou, 
    GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019. 
    Available from http://gailgithub.github.io/GAIL_Dev/
"""

from numpy import *

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

def gail_lattice_gen(n_min, n_max, d, z):
    """
    Generate d dimensionsal lattice samples from n_min to n_max
    
    Args:
        d (int): dimension of the problem.
        n_min (int): minimum index.
        n_max (int): maximum index (not inclusive). 
        z (int): length d generating vector.
    
    Returns:
        ndarray: n samples by d dimensions array of lattice samples
    """
    m_low = floor(log2(n_min))+1 if n_min > 0 else 0
    m_high = ceil(log2(n_max))
    x_lat_full = vstack([gen_block(m,z) for m in range(int(m_low),int(m_high)+1)])
    cut1 = int(floor(n_min-2**(m_low-1))) if n_min>0 else 0
    cut2 = int(cut1+n_max-n_min)
    x_lat = x_lat_full[cut1:cut2,:]
    return x_lat
