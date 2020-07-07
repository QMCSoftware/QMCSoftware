"""
Compare Monte Carlo and Quasi-Monte Carlo methods when
evaluating the Keister function with varying dimension
"""

from qmcpy import *
from workouts.mc_vs_qmc.integrations_keister import integrations_dict
from time import time
from numpy import arange, nan
import pandas as pd


def vary_dimension(dimension=[1,2,3], abs_tol=0, rel_tol=.1, trials=1):
    """
    Record solution, wall-clock time, and number of samples
    for integrating the Keister function with varying dimensions
    """
    header = ['Stopping Criterion','Distribution','MC/QMC','dimension','solution','n_samples','time']
    results = pd.DataFrame(columns=header)
    print(('%-20s'*2+'%-15s'*5)%tuple(header))
    i = 0
    for problem,function in integrations_dict.items():
        for dim in dimension:
            solution = 0
            n = 0
            time = 0
            for j in range(trials):
                data = function(dimension=dim, abs_tol=abs_tol, rel_tol=rel_tol)
                solution += data.solution
                n += data.n_total
                time += data.time_integrate
            results_i = [problem[0],problem[1],problem[2],dim,float(solution)/trials,float(n)/trials,float(time)/trials]
            results.loc[i] = results_i
            print(('%-20s%-20s%-15s%-15d%-15.2f%-15d%-15.3f')%tuple(results_i))
            i += 1
    return results


if __name__ == '__main__':
    results = vary_dimension(dimension=arange(1,41), abs_tol=0, rel_tol=.01, trials=3)
    results.to_csv('workouts/mc_vs_qmc/out/vary_dimension.csv', index=False)
    