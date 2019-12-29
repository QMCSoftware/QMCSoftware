""" Compare Sobol generatros with different backends """

from qmcpy import Sobol

from time import time
from pandas import DataFrame
from numpy import zeros


def comp_sobol_backend(sample_sizes, trials=1, dims=1, replications=2):
    # initialize DataFrame
    results = {
        'n': sample_sizes,
        'Sobol_MPS_time': zeros(len(sample_sizes)),
        'Sobol_Pytorch_time': zeros(len(sample_sizes))}
    for j in range(trials):
        for i, n in enumerate(sample_sizes):
            # Magic Point Shop Backend
            t0 = time()
            if i == 0:  # initialize generator
                sobol_mps = Sobol(rng_seed=7, backend='mps')
            x = sobol_mps.gen_dd_samples(replications, n, dims, scramble=True)
            results['Sobol_MPS_time'][i] += time() - t0
            # Pytorch Backend
            t0 = time()
            if i == 0:  # initialize generators
                sobol_pytorch = Sobol(rng_seed=7, backend='pytorch')
            x = sobol_pytorch.gen_dd_samples(replications, n, dims, scramble=True)
            results['Sobol_Pytorch_time'][i] += time() - t0
            # Set/Print Results for this n
            print('trial = %-5d n=%-10d done' % (j, int(n)))
    # take average times
    results['Sobol_MPS_time'] = results['Sobol_MPS_time'] / trials
    results['Sobol_Pytorch_time'] = results['Sobol_Pytorch_time'] / trials
    # convert to pandas DataFrame
    results_df = DataFrame(results, dtype=float)
    return results_df


if __name__ == '__main__':
    sample_sizes = [16] + [2**i for i in range(4, 22)]  # [16, 16, 32, 64, 128, ... 2^(21)]
    df_times = comp_sobol_backend(sample_sizes, trials=3, dims=4, replications=16)
    df_times.to_csv('outputs/lds_sequences/sobol_backend_times.csv', index=False)
    print('\n', df_times)
