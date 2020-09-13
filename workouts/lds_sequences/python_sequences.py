""" Record times for QMCPy's quasi-random generators """

from qmcpy import *
from time import time
from numpy import *
from pandas import DataFrame
import warnings


def python_sequences(powers_2=arange(1, 4), trials=1, dimension=1):
    """
    Record time for generating samples from each discrete distribution
    """
    print('\nDiscrete DiscreteDistribution Generation Time Comparison')
    columns = ['n',
        'L_MPS', 'L_GAIL',
        'S_gc', 'S_n',
        'H_QRNG', 'H_Owen',
        'K_QRNG']
    warnings.simplefilter('ignore')
    dds = [
        Lattice(dimension, randomize=True, seed=7, backend='MPS'),
        Lattice(dimension, randomize=True, seed=7, backend='GAIL'),
        Sobol(dimension, randomize=True, seed=7, graycode=True),
        Sobol(dimension, randomize=True, seed=7,  graycode=False),
        Halton(dimension, generalize=True, backend='QRNG', seed=7),
        Halton(dimension, generalize=True, backend='Owen', seed=7),
        Korobov(dimension, generator=[1], randomize=True)]
    df = DataFrame(columns=columns, dtype=float)
    for i, m in enumerate(powers_2):
        n = 2**m
        row_i = {'n': n}
        for j in range(len(columns)-1):
            dd_name = columns[j+1]
            dd = dds[j]
            t0 = time()
            for trial in range(trials):
                x = dd.gen_samples(n)
            row_i[dd_name] = (time() - t0) / trials
        df.loc[i] = row_i
        print('\n'.join(['%s: %.4f'%(key,val) for key,val in row_i.items()])+'\n')
    return df


if __name__ == '__main__':
    df_times = python_sequences(powers_2=arange(1, 21), trials=3, dimension=1)
    df_times.to_csv('workouts/lds_sequences/out/python_sequences.csv', index=False)
    print('\n', df_times)
