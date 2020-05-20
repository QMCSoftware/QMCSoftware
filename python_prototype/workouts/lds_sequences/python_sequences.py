""" Record times for QMCPy's quasi-random generators """

from qmcpy import *
from time import time
from numpy import *
from pandas import DataFrame


def python_sequences(powers_2=arange(1, 4), trials=1, dimension=1):
    """
    Record time for generating samples from each discrete distribution
    """
    print('\nDiscrete DiscreteDistribution Generation Time Comparison')
    columns = ['n', 'L_MPS_t', 'L_GAIL_t', 'S_QRNG_t', 'S_MPS_QMCPy_t', 'S_PyTorch_t']
    df = DataFrame(columns=columns, dtype=float)
    for i, m in enumerate(powers_2):
        n = 2**m
        row_i = {'n': n}
        # Lattice Magic Point Shop
        t0 = time()
        for trial in range(trials):
            distribution = Lattice(dimension, scramble=False, seed=7, backend='MPS')
            x = distribution.gen_samples(n_min=0,n_max=n)
        row_i['L_MPS_t'] = (time() - t0) / trials
        # Lattice GAIL
        t0 = time()
        for trial in range(trials):
            distribution = Lattice(dimension, scramble=False, seed=7, backend='GAIL')
            x = distribution.gen_samples(n_min=0,n_max=n)
        row_i['L_GAIL_t'] = (time() - t0) / trials
        # Sobol QRNG
        t0 = time()
        for trial in range(trials):
            distribution = Sobol(dimension, scramble=False, seed=7, backend='QRNG')
            distribution.gen_samples(n_min=0,n_max=n)
        row_i['S_QRNG_t'] = (time() - t0) / trials
        # Sobol Magic Point Shop QMCPy
        t0 = time()
        for trial in range(trials):
            distribution = Sobol(dimension, scramble=False, seed=7, backend='MPS')
            x = distribution.gen_samples(n_min=0,n_max=n)
        row_i['S_MPS_QMCPy_t'] = (time() - t0) / trials
        # Sobol PyTorch QMCPy
        t0 = time()
        for trial in range(trials):
            distribution = Sobol(dimension, scramble=False, seed=7, backend='PyTorch')
            x = distribution.gen_samples(n_min=0,n_max=n)
        row_i['S_PyTorch_t'] = (time() - t0) / trials
        # Set and print results
        df.loc[i] = row_i
        print('\n'.join(['%s: %.4f'%(key,val) for key,val in row_i.items()])+'\n')
    return df


if __name__ == '__main__':
    df_times = python_sequences(powers_2=arange(1, 21), trials=3, dimension=1)
    df_times.to_csv('outputs/lds_sequences/python_sequence_times.csv', index=False)
    print('\n', df_times)
