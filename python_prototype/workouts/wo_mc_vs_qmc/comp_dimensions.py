""" Comparing mc and qmc varying parameters """

from qmcpy import *

import json
from numpy import nan, zeros
from pandas import DataFrame

distribution_pointers = [IIDStdUniform, IIDStdGaussian, Lattice, Sobol]
trials = 3

def dimension_comparison(dimensions=arange(1, 4, 1)):
    """
    Record solution, wall-clock time, and number of samples
    for integrating the Keister function with varying dimensions
    """
    print('\nDimension Comparison')
    columns = ['dimension'] + \
        [type(distrib()).__name__ + '_solution' for distrib in distribution_pointers] + \
        [type(distrib()).__name__ + '_time' for distrib in distribution_pointers] + \
        [type(distrib()).__name__ + '_n' for distrib in distribution_pointers]
    df = DataFrame(columns=columns, dtype=float)
    for i, dimension in enumerate(dimensions):
        row_i = {'dimension': dimension}
        for distrib_pointer in distribution_pointers:
            for j in range(trials):
                sols, times, ns = zeros(trials), zeros(trials), zeros(trials)
                distribution = distrib_pointer(rng_seed=7)
                integrand = Keister(dimension=dimension)
                measure = Gaussian(dimension=dimension, variance=1 / 2)
                distrib_name = type(distribution).__name__
                if distrib_name in ['IIDStdGaussian', 'IIDStdUniform']:
                    stopping_criterion = CLT(distribution, measure, rel_tol=.01, abs_tol=0,
                                            n_max=1e10, n_init=256)
                elif distrib_name in ['Lattice', 'Sobol']:
                    stopping_criterion = CLTRep(distribution, measure, rel_tol=.01, abs_tol=0,
                                                n_max=1e10, n_init=32)
                try:
                    sol, data = integrate(integrand, measure, distribution, stopping_criterion)
                    sols[j] = sol
                    times[j] = data.time_total
                    ns[j] = data.n_total
                except:
                    sols[j] = nan
                    times[j] = nan
                    ns[j] = nan
                row_i[distrib_name + '_solution'] = sols.mean()
            row_i[distrib_name + '_time'] = times.mean()
            row_i[distrib_name + '_n'] = ns.mean()
        print(row_i)
        df.loc[i] = row_i
    return df


if __name__ == '__main__':    
    # Dimension Comparison Test
    dimensions = arange(1, 41)
    df_dimensions = dimension_comparison(dimensions)
    df_dimensions.to_csv('outputs/mc_vs_qmc/dimension.csv', index=False)
