from time import time
from numpy import *

from qmcpy.third_party.magic_point_shop import LatticeSeq
from qmcpy.discrete_distribution import DigitalSeq

def gen_lattice_points(n,d):
    t0 = time()
    lattice_rng = LatticeSeq(m=30,s=d)
    x = array([next(lattice_rng) for i in range(n)])
    print('Lattice Time: %.3f'%(time()-t0))
    return x

def gen_sobol_points(n,d):
    t0 = time()
    sobol_rng = DigitalSeq(Cs="sobol_Cs.col", m=30, s=d)
    x = zeros((n, d), dtype=int64)
    for i in range(n):
        next(sobol_rng)
        x[i, :] = sobol_rng.cur
    print('Sobol Time: %.3f'%(time()-t0))
    return x

n = 2**21
d = 1
gen_lattice_points(n,d)
gen_sobol_points(n,d)
