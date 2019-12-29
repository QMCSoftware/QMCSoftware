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

from time import time

from numpy import *
from pandas import DataFrame
# refactored generators
from qmcpy.discrete_distribution.mps_refactor import DigitalSeq, LatticeSeq
# origianl generators
from third_party.magic_point_shop import digitalseq_b2g, latticeseq_b2


def mps_gentimes(n_2powers=arange(1, 11), check_accuracy=False, trials=1, dim=1):
    """
    Record time for generating samples from
    original and modified Magic Point Shop generators
    """
    print('\nMagic Point Shop Generation Time Comparison')
    columns = ['n'] + \
        ['mps_lattice_time', 'qmcpy_lattice_time'] +\
        ['mps_Sobol_time', 'qmcpy_Sobol_time']
    df = DataFrame(columns=columns, dtype=float)
    for n_2 in n_2powers:
        n_samples = 2**n_2
        row_i = {'n': n_samples}
        # Original MPS Lattice
        t0 = time()
        for j in range(trials):
            lattice_rng = latticeseq_b2(m=30, s=dim, returnDeepCopy=True)
            mps_lattice_samples = array(
                [next(lattice_rng) for i in range(n_samples)])
        row_i['mps_lattice_time'] = (time() - t0) / trials
        # Refactored MPS Lattice
        t0 = time()
        for j in range(trials):
            lattice_rng = LatticeSeq(m=30, s=dim, returnDeepCopy=False)
            qmcpy_lattice_samples = vstack(
                [lattice_rng.calc_block(m) for m in range(n_2 + 1)])
        row_i['qmcpy_lattice_time'] = (time() - t0) / trials
        # Lattice Accuracy Check
        if check_accuracy and not all(row in qmcpy_lattice_samples for row in mps_lattice_samples):
            raise Exception(
                "Lattice sample do not match for n_samples=2^%d" % n_2)
        # Original MPS Sobol
        t0 = time()
        for j in range(trials):
            sobol_rng = digitalseq_b2g(Cs="./third_party/magic_point_shop/sobol_Cs.col", m=30, s=dim, returnDeepCopy=True)
            mps_Sobol_samples = zeros((n_samples, dim), dtype=int64)
            for i in range(n_samples):
                next(sobol_rng)
                mps_Sobol_samples[i, :] = sobol_rng.cur
        row_i['mps_Sobol_time'] = (time() - t0) / trials
        # Refactored MPS Sobol
        t0 = time()
        for j in range(trials):
            sobol_rng = DigitalSeq(Cs="sobol_Cs.col", m=30, s=dim)
            qmcpy_Sobol_samples = zeros((n_samples, dim), dtype=int64)
            for i in range(n_samples):
                next(sobol_rng)
                qmcpy_Sobol_samples[i, :] = sobol_rng.cur
        row_i['qmcpy_Sobol_time'] = (time() - t0) / trials
        # Sobol Accuracy Check
        if check_accuracy and not all(row in qmcpy_Sobol_samples for row in mps_Sobol_samples):
            raise Exception(
                "Sobol sample do not match for n_samples=2^%d" % n_2)
        # Set/Print Results for this n
        print(row_i)
        df.loc[i] = row_i
    return df


if __name__ == '__main__':
    df_times = mps_gentimes(n_2powers=arange(
        1, 21), check_accuracy=True, trials=3, dim=1)
    df_times.to_csv(
        'outputs/lds_sequences/magic_point_shop_times.csv', index=False)
    print('\n', df_times)
