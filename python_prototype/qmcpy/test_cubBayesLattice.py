import sys
sys.path.append("F:\\git-repos\\QMCSoftware\\python_prototype")
from qmcpy import *


dim = 3
integrand = Keister(dim)
true_measure = Gaussian(dim, variance=1 / 2)
discrete_distrib = Sobol(rng_seed=7)
stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=0.05)
_, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
print(data)
