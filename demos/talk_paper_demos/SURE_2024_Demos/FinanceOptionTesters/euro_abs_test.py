# Created: July 11, 2024
# Author: Richard Varela
# Email: richardvare13@gmail.com
# Purpose: To test the European option class to make sure it still works correctly
# 	after refactoring it to use the abstract Option class.

import numpy as np
import qmcpy as qp
import matplotlib.pyplot as plt

dims = 16
abs_tol = 1e-3

aco = qp.EuropeanOption(qp.Sobol(dims), volatility=0.2, start_price=100, strike_price=120, interest_rate=.05, call_put='call')
kde, a, b, approx_solution, data, fig, axes = qp.CubQMCSobolG(aco, abs_tol).density_estimation(True)
print("European Option value: %.5f (to within %1.e) with %.f paths"% (approx_solution, abs_tol, data.n_total))

plt.show()
