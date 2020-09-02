from ...util import ParameterError, ParameterWarning
from numpy import *
import os
import warnings


class LatticeGAIL(object):
    """
    Lattice Generator from GAIL (http://gailgithub.github.io/GAIL_Dev/)

    Original Implementations:

        https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/%2Bgail/vdc.m
        
        https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/%2Bgail/lattice_gen.m

    References:

        [1] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
        Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou, 
        GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019. 
        Available from http://gailgithub.github.io/GAIL_Dev/
    """

    def __init__(self, dimension, randomize, seed, gen_vector_info):
        if gen_vector_info:
            self.z_full = array(gen_vector_info['vector'], dtype=double)
            self.n_lim = gen_vector_info['n_lim']
        else:
            abs_file_path = os.path.join(os.path.dirname(__file__),
                                         'lattice-32001-1024-1048576.3600.npy')
            self.z_full = load(abs_file_path).astype(double)
            self.n_lim = 2**20
        self.d_lim = len(self.z_full)
        self.r = randomize
        self.set_dimension(dimension)
        self.set_seed(seed)

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
    
    def gen_samples(self,n_min, n_max, warn, linear=False, return_non_rand=False):
        if n_min == 0 and self.r==False and warn:
            warnings.warn("Non-randomized lattice sequence includes the origin",ParameterWarning)
        if n_max > self.n_lim:
            raise ParameterError('Lattice generating vector supports up to %d samples.'%self.n_lim)
        m_low = floor(log2(n_min)) + 1 if n_min > 0 else 0
        m_high = ceil(log2(n_max))
        if linear:
            nelem = n_max - n_min
            if n_min == 0:
                y = arange(0, 1, 1 / nelem).reshape((nelem, 1))
            else:
                y = arange(1 / n_max, 1, 2 / n_max).reshape((nelem, 1))
            x = outer(y, self.z) % 1
        else:
            x_lat_full = vstack([self._gen_block(m) for m in range(int(m_low),int(m_high)+1)])
            cut1 = int(floor(n_min-2**(m_low-1))) if n_min>0 else 0
            cut2 = int(cut1+n_max-n_min)
            x = x_lat_full[cut1:cut2,:]
        if self.r: # apply random shift to samples
            x_rand = (x + self.shift)%1
            if return_non_rand:
                return x_rand, x
            else:
                return x_rand
        else:
            return x

    def set_seed(self, seed):
        self.s = seed if seed else random.randint(2**32)
        random.seed(self.s)
        if self.r:
            self.shift = random.rand(int(self.d))
        return self.s
        
    def set_dimension(self, dimension):
        self.d = dimension
        if self.d > self.d_lim:
            raise ParameterError('GAIL Lattice requires dimension <= %d'%self.d_lim)
        self.z = self.z_full[:self.d]
        if self.r:
            self.shift = random.rand(int(self.d))
        return self.d
    
    def get_params(self):
        return self.d, self.r, self.s