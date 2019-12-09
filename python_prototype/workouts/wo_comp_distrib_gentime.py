from qmcpy import *

from numpy import *
from pandas import DataFrame
from time import time

distribution_pointers = [IIDStdUniform, IIDStdGaussian, Lattice, Sobol]

def samples_gentime_comparison(n_2powers=arange(1,11)):
    """
    Record wall-clock time for generating samples from each discrete distribution
    """
    dimension = 1
    print('\nDiscrete Distribution Generation Time Comparison')
    columns = ['n_2power'] + [type(distrib()).__name__ + '_time' for distrib in distribution_pointers]
    df = DataFrame(columns=columns, dtype=float)
    for i, n_2 in enumerate(n_2powers):
        n_samples = 2**n_2
        row_i = {'n_2power': n_2}
        for distrib_pointer in distribution_pointers:
            t0 = time()
            distribution = distrib_pointer(rng_seed=7)
            distribution.gen_dd_samples(1, n_samples, dimension)
            row_i[type(distribution).__name__ + '_time'] = time()-t0
        print(row_i)
        df.loc[i] = row_i
    return df

if __name__ == '__main__': 
    df = samples_gentime_comparison(n_2powers=arange(1,21))
    print('\n',df)