"""
Examples of Lattice and Sobol Sequences.
    Run Example:
        python workouts/wo_sequences.py
    Save Output:
        python workouts/wo_sequences.py  > outputs/examples/wo_sequences.txt
"""

from qmcpy import *
from third_party.magic_point_shop import digitalseq_b2g, latticeseq_b2

from time import time
from numpy import *

def gen_mps_lattice_points(n, d):
    t0 = time()
    lattice_rng = latticeseq_b2(m=30, s=d)
    x = array([next(lattice_rng) for i in range(n)])
    t1 = time() - t0
    return x, t1

def gen_qmcpy_lattice_points(n, d):
    t0 = time()
    lattice = Lattice()
    x = lattice.gen_dd_samples(replications=1, n_samples=n, dimensions=d)
    t1 = time() - t0
    return x, t1

def gen_mps_sobol_points(n, d, Cs = "./third_party/magic_point_shop/sobol_Cs.col"):
    t0 = time()
    m = math.log(n, 2)
    sobol_rng = digitalseq_b2g(Cs=Cs, m=m, s=d)
    x = zeros((n, d), dtype=int64)
    for i in range(n):
        next(sobol_rng)
        x[i, :] = sobol_rng.cur
    t1 = time() - t0
    return x, t1

def gen_qmcpy_sobol_points(n, d):
    t0 = time()
    sobol = Sobol()
    x = sobol.gen_dd_samples(replications=1, n_samples=n, dimensions=d)
    t1 = time() - t0
    return x, t1

if __name__ == "__main__":
    m = 5
    n = 2 ** m
    d = 4
    print("n=%d, m=%d, d=%d" % (n, m, d))
    # Lattice
    mps_lattice_samples, mps_lattice_time = gen_mps_lattice_points(n, d)
    print('\nMPS Lattice Time: %.3f' % mps_lattice_time)
    qmcpy_lattice_samples, qmcpy_lattice_time = gen_qmcpy_lattice_points(n, d) 
    print('\nQMCPy Lattice Time: %.3f' % qmcpy_lattice_time)
    # Sobol
    mps_sobol_samples, mps_sobol_time = gen_mps_sobol_points(n, d)
    print('\nMPS Sobol Time: %.3f' % mps_sobol_time)
    qmcpy_sobol_samples, qmcpy_sobol_time = gen_qmcpy_sobol_points(n, d)
    print('\nQMCPy Sobol Time: %.3f' % qmcpy_sobol_time)
