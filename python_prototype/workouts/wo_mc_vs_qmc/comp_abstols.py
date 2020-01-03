""" Comparing mc and qmc varying parameters """

from numpy import arange, nan, zeros
from pandas import DataFrame
from qmcpy import *

distribution_pointers = [IIDStdUniform, IIDStdGaussian, Lattice, Sobol]
trials = 3


def abstol_comparison(abstols=arange(.1, .4, .1)):
    """
    Record solution, wall-clock time, and number of samples
    for integrating the Keister function with
    varying absolute tolerances
    """
    dimension = 3
    print('\nAbsolute Tolerance Comparison')
    columns = ['abs_tol'] + \
        [type(distrib()).__name__ + '_solution' for distrib in distribution_pointers] + \
        [type(distrib()).__name__ + '_time' for distrib in distribution_pointers] + \
        [type(distrib()).__name__ + '_n' for distrib in distribution_pointers]
    df = DataFrame(columns=columns, dtype=float)
    for i, abs_tol in enumerate(abstols):
        row_i = {'abs_tol': abs_tol}
        for distrib_pointer in distribution_pointers:
            for j in range(trials):
                sols, times, ns = zeros(trials), zeros(trials), zeros(trials)
                distribution = distrib_pointer(rng_seed=7)
                integrand = Keister(dimension=dimension)
                measure = Gaussian(dimension=dimension, variance=1 / 2)
                distrib_name = type(distribution).__name__
                if distrib_name in ['IIDStdGaussian', 'IIDStdUniform']:
                    stopping_criterion = CLT(distribution, measure, abs_tol=abs_tol,
                                             n_max=1e10, n_init=256)
                elif distrib_name in ['Lattice', 'Sobol']:
                    stopping_criterion = CLTRep(distribution, measure,
                                                abs_tol=abs_tol, n_max=1e10,
                                                n_init=32)
                try:
                    sol, data = integrate(integrand, measure,
                                          distribution, stopping_criterion)
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
    # Absolute Tolerance Comparison Test
    abstols = arange(.001, .0502, .0002)
    df_abstols = abstol_comparison(abstols)
    df_abstols.to_csv('outputs/mc_vs_qmc/abs_tol.csv', index=False)
