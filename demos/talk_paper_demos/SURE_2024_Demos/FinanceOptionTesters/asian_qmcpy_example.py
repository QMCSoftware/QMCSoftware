# Created: June 20, 2024
# Author: Richard Varela
# Email: richardvare13@gmail.com
# Purpose: Used to test the Asian option after refactoring it to use the abstract Option class.
# 	This code uses an example from QMCPy documentation: https://qmcpy.readthedocs.io/en/latest/demo_rst/asian-option-mlqmc.html

import numpy as np
import qmcpy as qp
import matplotlib.pyplot as plt

seed = 7

def eval_option(option_mc, option_qmc, abs_tol):
    stopping_criteria = {
        "MLMC" : qp.CubMCML(option_mc, abs_tol=abs_tol, levels_max=15),
        "continuation MLMC" : qp.CubMCMLCont(option_mc, abs_tol=abs_tol, levels_max=15),
        "MLQMC" : qp.CubQMCML(option_qmc, abs_tol=abs_tol, levels_max=15),
        "continuation MLQMC" : qp.CubQMCMLCont(option_qmc, abs_tol=abs_tol, levels_max=15)
    }

    levels = []
    times = []
    for name, stopper in stopping_criteria.items():
        sol, data = stopper.integrate()
        levels.append(data.levels)
        times.append(data.time_integrate)
        print("\t%-20s solution %-10.4f number of levels %-6d time %.3f"%(name, sol, levels[-1], times[-1]))

    return levels, times

# Body
# Part 1
for level in range(5):
    aco = qp.AsianOption(qp.Sobol(2*2**level, seed=seed), volatility=.2, start_price=100, strike_price=100, interest_rate=.05)
    approx_solution, data = qp.CubQMCSobolG(aco, abs_tol=1e-4).integrate()
    print("Asian Option true value (%d time steps): %.5f (to within 1e-4)"%(2*2**level, approx_solution))

# Part 2
option_mc = qp.MLCallOptions(qp.IIDStdUniform(seed=seed), option="asian")
option_qmc = qp.MLCallOptions(qp.Lattice(seed=seed), option="asian")

eval_option(option_mc, option_qmc, abs_tol=5e-3)

repetitions = 5
tolerances = 5*np.logspace(-1, -3, num=5)

levels = {}
times = {}
for t in range(len(tolerances)):
    for r in range(repetitions):
        print("tolerance = %10.4e, repetition = %d/%d"%(tolerances[t], r + 1, repetitions))
        levels[t, r], times[t, r] = eval_option(option_mc, option_qmc, tolerances[t])

avg_time = {}
for method in range(4):
    avg_time[method] = [np.mean([times[t, r][method] for r in range(repetitions)]) for t in range(len(tolerances))]

plt.figure(figsize=(10,7))
plt.plot(tolerances, avg_time[0], label="MLMC")
plt.plot(tolerances, avg_time[1], label="continuation MLMC")
plt.plot(tolerances, avg_time[2], label="MLQMC")
plt.plot(tolerances, avg_time[3], label="continuation MLQMC")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("requested absolute tolerance")
plt.ylabel("average run time in seconds")
plt.legend();
