# Created: June 11, 2024
# Author: Richard Varela
# Email: richardvare13@gmail.com

import numpy as np
import qmcpy as qp
import matplotlib.pyplot as plt

dims = 16
abs_tol = 1e-3

# aco = qp.LookBackOption(qp.Sobol(1), volatility=0.2, start_price=100, interest_rate=.05, call_put='call')
aco = qp.LookBackOption(volatility=0.2, start_price=100, interest_rate=.05, call_put='put')
# kde, a, b, approx_solution, data = qp.CubQMCSobolG(aco, abs_tol).density_estimation()
kde, a, b, approx_solution, data, fig, ax1 = qp.CubQMCSobolG(aco, abs_tol).density_estimation(True)
# breakpoint()
print(f"Discounted payoffs: {aco.get_discounted_payoffs()}")
x = aco.get_discounted_payoffs()
print(x)
# print(type(x))
# print(f"Max: {x.max()}")
# print(f"Min: {x.min()}")
print("Lookback Option value: %.5f (to within %1.e) with %.f paths"% (approx_solution, abs_tol, data.n_total))
plt.show()

