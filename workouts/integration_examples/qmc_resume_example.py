"""
Example: Using the resume feature in QMC integration
"""
from qmcpy import CubQMCLatticeG, Genz, Gaussian, Lattice
import numpy as np

# Define integrand and measure
discrete_distrib = Lattice(dimension=3)
true_measure = Gaussian(discrete_distrib, mean=0, covariance=1)

# Use Genz integrand (oscillatory kind)
integrand = Genz(discrete_distrib, kind_func='oscillatory', kind_coeff=1)

# First run: loose tolerance, stop early
abs_tol = 1e-3
rel_tol = 0
solver = CubQMCLatticeG(integrand, abs_tol=abs_tol, rel_tol=rel_tol)
solution1, data1 = solver.integrate()
print(f"First run solution: {solution1}")

# Second run: tighter tolerance, resume from previous data
abs_tol = 1e-5
solver.set_tolerance(abs_tol=abs_tol)
solution2, data2 = solver.integrate(resume=data1)
print(f"Resumed run solution: {solution2}")

# Third run: tighter tolerance, start from scratch (no resume)
abs_tol = 1e-5
solver = CubQMCLatticeG(integrand, abs_tol=abs_tol, rel_tol=rel_tol)
solution3, data3 = solver.integrate()
print(f"Second run without resume solution: {solution3}")
