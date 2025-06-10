"""
Example: Saving and loading QMC integration state with resume feature
"""
from qmcpy import CubQMCLatticeG, Genz, Gaussian, Lattice
from qmcpy.accumulate_data._accumulate_data import AccumulateData
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

# Save the data to disk after first run
save_path = 'resume_data_example2.pkl'
data1.save(save_path)
print(f"Saved first run data to {save_path}")

# Later, or in a new session, load the data and resume with tighter tolerance
loaded_data = AccumulateData.load(save_path)
abs_tol = 1e-5
solver.set_tolerance(abs_tol=abs_tol)
solution2, data2 = solver.integrate(resume=loaded_data)
print(f"Resumed from disk with tighter tolerance solution: {solution2}")

# Optionally, save the resumed data again
save_path2 = 'resume_data_example2_resumed.pkl'
data2.save(save_path2)
print(f"Saved resumed data to {save_path2}")

# Start from scratch for comparison
solver = CubQMCLatticeG(integrand, abs_tol=abs_tol, rel_tol=rel_tol)
solution3, data3 = solver.integrate()
print(f"Run from scratch with tight tolerance solution: {solution3}")
