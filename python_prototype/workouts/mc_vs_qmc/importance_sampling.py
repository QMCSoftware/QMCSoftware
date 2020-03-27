""" Importance Sampling Tests """

from numpy import *
import pandas as pd
from workouts.mc_vs_qmc.integrations_asian_call import integrations_dict


def compare_mean_shifts(abs_tol, dimension, time_vector, mean_shifts, trials):
    header = ['Stopping Criterion','Distribution','mean_shift','n_samples','time']
    results = pd.DataFrame(columns=header)
    print('%-20s%-15s%-15s%-15s%-15s'%tuple(header))
    i = 0
    for problem,function in integrations_dict.items():
        for ms in mean_shifts:
            n = 0
            time = 0
            for trial in range(trials):
                data = function(dimension, abs_tol, time_vector, ms)
                n += data.n_total
                time += data.time_integrate
            n = float(n)/trials
            time = float(time)/trials
            results_i = [problem[0],problem[1],ms,n,time]
            results.loc[i] = results_i
            print('%-20s%-15s%-15.2f%-15d%-15.3f'%tuple(results_i))
            i += 1
    return results


if __name__ == '__main__':
    dimension = 16
    tv = [i/dimension for i in range(1,dimension+1)]
    df_cms = compare_mean_shifts(
        abs_tol = .025,
        dimension = dimension,
        time_vector = tv,
        mean_shifts = arange(-.25,3.25,.25),
        trials = 3)
    print(df_cms.head())
    df_cms.to_csv('outputs/mc_vs_qmc/importance_sampling_compare_mean_shifts.csv', index=False)
