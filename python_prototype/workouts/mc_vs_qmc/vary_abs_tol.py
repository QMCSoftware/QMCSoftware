"""
Compare Monte Carlo and Quasi-Monte Carlo methods when
evaluating the Keister function with varying absolute tolerance
"""

from qmcpy import *
from integrations import integrations_dict
from time import time
from numpy import arange, nan
from pandas import DataFrame


def vary_abs_tol(dimension=3, abs_tol=[.1,.2,.3], trials=1):
    """
    Record solution, wall-clock time, and number of samples
    for integrating the Keister function with
    varying absolute tolerances
    """
    print('\nAbsolute Tolerance Comparison')
    df_solution = DataFrame(nan,
        index = arange(len(abs_tol)), \
        columns = ['abs_tol']+list(integrations_dict.keys()),
        dtype = float)
    df_n_total = df_solution.copy()
    df_time = df_solution.copy()
    for i,tol in enumerate(abs_tol):
        row_solution = {'abs_tol': tol}
        row_n_total = {'abs_tol': tol}
        row_time = {'abs_tol': tol}
        for name, function in integrations_dict.items():
            try:
                solution = 0
                n_total = 0
                time_total = 0
                for j in range(trials):
                    data = function(dimension=dimension, abs_tol=abs_tol)
                    solution += data.solution
                    n_total += data.n_total
                    time_total += data.time_total
                row_solution[name] = solution / trials
                row_n_total[name] = n_total / trials
                row_time[name] = time_total / trials
            except: pass
        # Set and print results
        df_solution.loc[i] = row_solution
        df_n_total.loc[i] = row_n_total
        df_time.loc[i] = row_time
        print('\n'.join(['%s: %.4f'%(key,val) for key,val in row_time.items()])+'\n')
    return df_solution,df_n_total,df_time

if __name__ == '__main__':
    df_solution,df_n_total,df_time = vary_abs_tol(dimension=3, abs_tol=arange(.001, .0502, .0002), trials=3)
    df_solution.to_csv('outputs/mc_vs_qmc/vary_abs_tol_solution.csv', index=False)
    df_n_total.to_csv('outputs/mc_vs_qmc/vary_abs_tol_n_total.csv', index=False)
    df_time.to_csv('outputs/mc_vs_qmc/vary_abs_tol_time.csv', index=False)