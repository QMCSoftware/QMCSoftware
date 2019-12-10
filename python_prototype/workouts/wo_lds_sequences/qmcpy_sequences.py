""" Record times for qmcpy's quasi-random generators """

from qmcpy import * 

from numpy import *
from pandas import DataFrame
from time import time

dim = 1
distribution_pointers = [Lattice, Sobol]

def qmcpy_gentimes(n_2powers=arange(1,11)):
    """
    Record wall-clock time for generating samples from each discrete distribution
    """
    print('\nDiscrete Distribution Generation Time Comparison')
    columns = ['n_2power'] + [type(distrib()).__name__ + '_time' for distrib in distribution_pointers]
    df = DataFrame(columns=columns, dtype=float)
    for i, n_2 in enumerate(n_2powers):
        n_samples = 2**n_2
        row_i = {'n_2power': n_2}
        for distrib_pointer in distribution_pointers:
            t0 = time()
            distribution = distrib_pointer(rng_seed=7)
            x = distribution.gen_dd_samples(1, n_samples, dim)
            row_i[type(distribution).__name__ + '_time'] = time()-t0
        print(row_i)
        df.loc[i] = row_i
    return df

if __name__ == '__main__': 
    df_times = qmcpy_gentimes(n_2powers=arange(1,21))
    df_times.to_csv('outputs/lds_sequences/python_sequence_times.csv', index=False)