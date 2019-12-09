"""
Compare origianl and refactored backend generators for quasirandom sequences.

Original package found at /third_party/magic_point_shop/
    Reference:
        D. Nuyens, `The Magic Point Shop of QMC point generators and generating
        vectors.` MATLAB and Python software, 2018. Available from
        https://people.cs.kuleuven.be/~dirk.nuyens/
Refactored package found at /qmcpy/discrete_distribution/mps_refactor/

Note:
    A unittest for refactored generators is at /test/fasttests/test_discrete_distributions.py
"""

from third_party.magic_point_shop import digitalseq_b2g, latticeseq_b2 # origianl generators
from qmcpy.discrete_distribution.mps_refactor import LatticeSeq, DigitalSeq # refactored generators

from time import time
from numpy import *

def original_lattice(n, d):
    t0 = time()
    lattice_rng = latticeseq_b2(m=30, s=d)
    x = array([next(lattice_rng) for i in range(n)])
    t1 = time() - t0
    return x, t1

def refactored_lattice(n, d):
    t0 = time()
    lattice_rng = LatticeSeq(m=30, s=d)
    x = array([next(lattice_rng) for i in range(n)])
    t1 = time() - t0
    return x, t1

def original_Sobol(n, d):
    t0 = time()
    m = math.log(n, 2)
    sobol_rng = digitalseq_b2g(Cs="./third_party/magic_point_shop/sobol_Cs.col", m=30, s=d)
    x = zeros((n, d), dtype=int64)
    for i in range(n):
        next(sobol_rng)
        x[i, :] = sobol_rng.cur
    t1 = time() - t0
    return x, t1

def refactored_Sobol(n, d):
    t0 = time()
    sobol_rng = DigitalSeq(Cs="sobol_Cs.col", m=30, s=d)
    x = zeros((n, d), dtype=int64)
    for i in range(n):
        next(sobol_rng)
        x[i, :] = sobol_rng.cur
    t1 = time() - t0
    return x, t1

if __name__ == "__main__":
    m = 15
    n = 2 ** m
    d = 4
    print("'\nn=%d, m=%d, d=%d\n" % (n, m, d))
    # Lattice
    mps_lattice_samples, mps_lattice_time = original_lattice(n, d)
    print('MPS Lattice Time: %.3f' % mps_lattice_time)
    qmcpy_lattice_samples, qmcpy_lattice_time = refactored_lattice(n, d) 
    print('QMCPy Lattice Time: %.3f' % qmcpy_lattice_time)
    # Sobol
    mps_sobol_samples, mps_sobol_time = original_Sobol(n, d)
    print('MPS Sobol Time: %.3f' % mps_sobol_time)
    qmcpy_sobol_samples, qmcpy_sobol_time = refactored_Sobol(n, d)
    print('QMCPy Sobol Time: %.3f' % qmcpy_sobol_time)
