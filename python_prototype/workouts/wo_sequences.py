"""
Examples of Lattice and Sobol Sequences.
    Run Example:
        python workouts/wo_sequences.py
    Save Output:
        python workouts/wo_sequences.py  > outputs/examples/wo_sequences.txt
"""

from time import time

from numpy import *
import numpy.testing as npt
from qmcpy import *
from qmcpy.third_party.magic_point_shop import digitalseq_b2g, LatticeSeq

returnDeepCopy = False


def gen_lattice_points(n, d):
    t0 = time()
    lattice_rng = LatticeSeq(m=30, s=d)
    x = array([next(lattice_rng) for i in range(n)])
    print('\nLattice Time: %.3f' % (time() - t0))
    return x


def gen_mps_sobol_points(n, d, Cs = "./qmcpy/third_party/magic_point_shop/sobol_Cs.col"):
    t0 = time()
    m = math.log(n, 2)
    sobol_rng = digitalseq_b2g(Cs=Cs, m=m, s=d)
    x = zeros((n, d), dtype=int64)
    for i in range(n):
        next(sobol_rng)
        x[i, :] = sobol_rng.cur
    t1 = time() - t0
    print('\nMPS Sobol Time: %.3f' % t1)
    return x, t1


def gen_qmcpy_sobol_points(n, d):
    t0 = time()
    sobol = Sobol()
    x = sobol.gen_dd_samples(replications=1, n_samples=n, dimensions=d,
                             returnDeepCopy=returnDeepCopy, scramble=False)
    t1 = time() - t0
    print('\nQMCPy Sobol Time: %.3f' % t1)
    return x, t1


if __name__ == "__main__":
    m = 21
    n = 2 ** m
    d = 4
    print("n=%d, m=%d, d=%d" % (n, m, d))
    gen_lattice_points(n, d)
    mps_sobol_samples, mps_time = gen_mps_sobol_points(n, d)
    qmcpy_mps_sobol_samples, qmcpy_time = gen_qmcpy_sobol_points(n, d)
    npt.assert_array_equal(mps_sobol_samples, qmcpy_mps_sobol_samples)
