from ...util import ParameterError, ParameterWarning
from numpy import *
import os
import warnings


class LatticeMPS(object):
    """ 
    Lattice sequence generator from Magic Point Shop (https://people.cs.kuleuven.be/~dirk.nuyens/qmc-generators/)

    Original Implementation:

        https://bitbucket.org/dnuyens/qmc-generators/src/master/python/latticeseq_b2.py

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
    
    def gen_samples(self, n_min, n_max, warn, linear=False, return_non_ramdom=False):
        if n_min == 0 and self.r==False and warn:
            warnings.warn("Non-randomized lattice sequence includes the origin",ParameterWarning)
        if n_max > self.n_lim:
            raise ParameterError('Lattice generating vector supports up to %d samples.'%self.n_lim)
        if return_non_ramdom:
            raise ParameterError('return_non_ramdom=True option not implemented.')
        if linear:
            raise ParameterError('linear=True option not implemented.')
        m_low = floor(log2(n_min))+1 if n_min > 0 else 0
        m_high = ceil(log2(n_max))
        gen_block = lambda n: (outer(arange(1, n+1, 2), self.z) % n) / float(n)
        x_lat_full = vstack([gen_block(2**m) for m in range(int(m_low),int(m_high)+1)])
        cut1 = int(floor(n_min-2**(m_low-1))) if n_min>0 else 0
        cut2 = int(cut1+n_max-n_min)
        x = x_lat_full[cut1:cut2,:]
        if self.r: # apply random shift to samples
            x = (x + self.shift)%1
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
