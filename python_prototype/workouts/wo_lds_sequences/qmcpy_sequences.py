""" Record times for qmcpy's quasi-random generators """

from qmcpy import *

from numpy import *
from pandas import DataFrame
from time import time
import torch

def qmcpy_gentimes(n_2powers=arange(1, 11), trials=1, dim=1):
    """
    Record time for generating samples from each discrete distribution
    """
    print('\nDiscrete Distribution Generation Time Comparison')
    columns = ['n', 'Lattice_time', 'Sobol_MPS_time', 'Sobol_Torch_time']
    df = DataFrame(columns=columns, dtype=float)
    for i, n_2 in enumerate(n_2powers):
        n_samples = 2**n_2
        row_i = {'n': n_samples}
        # Lattice
        t0 = time()
        for trial in range(trials):
            distrib = Lattice(rng_seed=7)
            x = distrib.gen_dd_samples(1, n_samples, dim)
        row_i['Lattice_time'] = (time()-t0) / trials
        # Sobol Magic Point Shop
        t0 = time()
        for trial in range(trials):
            distrib = Sobol(rng_seed=7, backend='MPS')
            x = distrib.gen_dd_samples(1, n_samples, dim)
        row_i['Sobol_MPS_time'] = (time()-t0) / trials
        # Sobol Pytorch
        t0 = time()
        for trial in range(trials):
            distrib = Sobol(rng_seed=7, backend='Pytorch')
            x = distrib.gen_dd_samples(1, n_samples, dim)
        row_i['Sobol_Torch_time'] = (time()-t0) / trials
        # Set/Print Results for this n
        print(row_i)
        df.loc[i] = row_i
    return df


if __name__ == '__main__':
    df_times = qmcpy_gentimes(n_2powers=arange(1, 21), trials=3, dim=1)
    df_times.to_csv('outputs/lds_sequences/python_sequence_times.csv', index=False)
    print('\n', df_times)