"""
Compare Monte Carlo and Quasi-Monte Carlo methods when
evaluating the Keister function with varying absolute tolerance
"""

from qmcpy import *
from workouts.mc_vs_qmc.integrations_keister import integrations_dict
from time import time
from numpy import arange, nan
import pandas as pd


def vary_abs_tol(dimension=3, abs_tol=[.1,.2,.3], rel_tol=0, trials=1):
    """
    Record solution, wall-clock time, and number of samples
    for integrating the Keister function with
    varying absolute tolerances
    """
    header = ['Stopping Criterion','Distribution','MC/QMC','abs_tol','solution','n_samples','time']
    results = pd.DataFrame(columns=header)
    print(('%-20s'*2+'%-15s'*5)%tuple(header))
    i = 0
    for problem,function in integrations_dict.items():
        for tol in abs_tol:
            try:               
                solution = 0
                n = 0
                time = 0
                for j in range(trials):
                    data = function(dimension=dimension, abs_tol=tol, rel_tol=rel_tol)
                    solution += data.solution
                    n += data.n_total
                    time += data.time_integrate
                results_i = [problem[0],problem[1],problem[2],tol,float(solution)/trials,float(n)/trials,float(time)/trials]
                results.loc[i] = results_i
                print(('%-20s%-20s%-15s%-15.3f%-15.2f%-15d%-15.3f')%tuple(results_i))
                i += 1
            except: pass
    return results

if __name__ == '__main__':
    results = vary_abs_tol(dimension=3, abs_tol=arange(.002, .1002, .0002), rel_tol=0, trials=3)
    results.to_csv('workouts/mc_vs_qmc/out/vary_abs_tol.csv', index=False)