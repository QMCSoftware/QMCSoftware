""" Importance Sampling Tests """

from numpy import *
import pandas as pd
from workouts.mc_vs_qmc.integrations_asian_call import integrations_dict


def compare_mean_shifts(abs_tol=.1, dimension=4, mean_shifts=[0,.5,1], trials=1):
    header = ['Stopping Criterion','Distribution','MC/QMC','mean_shift','solution','n_samples','time']
    results = pd.DataFrame(columns=header)
    print(('%-15s'*7)%tuple(header))
    i = 0
    for problem,function in integrations_dict.items():
        for ms in mean_shifts:
            solution = 0
            n = 0
            time = 0
            for trial in range(trials):
                data = function(dimension, abs_tol, ms)
                solution += data.solution
                n += data.n_total
                time += data.time_integrate
            results_i = [problem[0],problem[1],problem[2],ms,float(solution)/trials,float(n)/trials,float(time)/trials]
            results.loc[i] = results_i
            print(('%-15s'*3+'%-15.2f%-15.2f%-15d%-15.3f')%tuple(results_i))
            i += 1
    return results


if __name__ == '__main__':
    df_cms = compare_mean_shifts(
        abs_tol = .025,
        dimension = 16,
        mean_shifts = around(arange(-.2,3.1,.1),decimals=1),
        trials = 3)
    df_cms.to_csv('workouts/mc_vs_qmc/out/importance_sampling.csv', index=False)
