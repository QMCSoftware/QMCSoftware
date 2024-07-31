# Created: June 20, 2024
# Author: Richard Varela
# Email: richardvare13@gmail.com

import numpy as np
import qmcpy as qp
import matplotlib.pyplot as plt

dims = 2**4
abs_tol = 1e-3
obs = 2**3

aco = qp.AmericanOption(sampler=qp.Sobol(obs), volatility=0.2, start_price=100, strike_price=150, interest_rate=0.05, n=obs, call_put='call', observations=obs)
# aco = qp.AmericanOption(volatility=0.2, start_price=100, strike_price=150, interest_rate=0.05, call_put='call', observations=4)
# breakpoint()
kde, a, b, approx_solution, data, fig, axes = qp.CubQMCSobolG(aco, abs_tol).density_estimation(True)
# kde, a, b, approx_solution, data = qp.CubQMCSobolG(aco, abs_tol).density_estimation()
print("American Option value: %.5f (to within %1.e) with %.f paths"% (approx_solution, abs_tol, data.n_total))

# fig.show()
plt.show()

# print("Discounted payoffs")
# print(aco.get_discounted_payoffs())
# x = aco._get_discounted_payoffs()
# print(type(x))
# print(x)
# print(f"Size of values: {x.size}")
# print(f"Max: {x.max()}")
# print(f"Min: {x.min()}")

# print("Output of density estimation")
# print(f"a = {a}, b = {b}")
# print(f"Approx solution: {approx_solution}")

