"""
Compare Monte Carlo and Quasi-Monte Carlo methods when
evaluating the Keister function with varying dimension
"""

from qmcpy import *
from workouts.mc_vs_qmc.integrations import integrations_dict
from time import time
from numpy import arange, nan
from pandas import DataFrame


def vary_dimension(dimension=[1,2,3], abs_tol=0, rel_tol=.1, trials=1):
    """
    Record solution, wall-clock time, and number of samples
    for integrating the Keister function with varying dimensions
    """
    print('\nDimension Comparison')
    df_solution = DataFrame(nan,
        index = arange(len(dimension)), \
        columns = ['dimension']+list(integrations_dict.keys()),
        dtype = float)
    df_n_total = df_solution.copy()
    df_time = df_solution.copy()
    for i,d in enumerate(dimension):
        row_solution = {'dimension': d}
        row_n_total = {'dimension': d}
        row_time = {'dimension': d}
        for name, function in integrations_dict.items():
            try:
                solution = 0
                n_total = 0
                time_integrate = 0
                for j in range(trials):
                    data = function(dimension=d, abs_tol=abs_tol, rel_tol=rel_tol)
                    solution += data.solution
                    n_total += data.n_total
                    time_integrate += data.time_integrate
                row_solution[name] = solution / trials
                row_n_total[name] = n_total / trials
                row_time[name] = time_integrate / trials
            except: pass
        # Set and print results
        df_solution.loc[i] = row_solution
        df_n_total.loc[i] = row_n_total
        df_time.loc[i] = row_time
        print('\n'.join(['%s: %.4f'%(key,val) for key,val in row_time.items()])+'\n')
    return df_solution,df_n_total,df_time


if __name__ == '__main__':
    df_solution,df_n_total,df_time = vary_dimension(dimension=arange(1, 41), abs_tol=0, rel_tol=.01, trials=3)
    df_solution.to_csv('outputs/mc_vs_qmc/vary_dimension_solution.csv', index=False)
    df_n_total.to_csv('outputs/mc_vs_qmc/vary_dimension_n_total.csv', index=False)
    df_time.to_csv('outputs/mc_vs_qmc/vary_dimension_time.csv', index=False)
    