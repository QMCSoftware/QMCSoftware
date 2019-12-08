# Wolfram Alpha: integral_(-2)^2 sqrt(4 - x^2) (1/2 + x^3 cos(x/2)) dx = 3.14159
# Answer looks to be pi

# Sseems that sobol is not working correctly 

from qmcpy import *
from numpy import *

custom_fun = lambda x: sqrt(4-x**2) * (1/2 + x**3 * cos(x/2))

# IID
integrand = QuickConstruct(
        dimension = 1,
        custom_fun = custom_fun)
true_measure = Lebesgue(dimension=1, lower_bound=-2, upper_bound=2)
discrete_distrib = IIDStdUniform(rng_seed=7)
stop_criterion = CLT(discrete_distrib, true_measure, abs_tol=.005)
sol,data_obj = integrate(integrand, true_measure, discrete_distrib, stop_criterion)
print(data_obj)

# Lattice 
integrand = QuickConstruct(
        dimension = 1,
        custom_fun = custom_fun)
true_measure = Lebesgue(dimension=1, lower_bound=-2, upper_bound=2)
discrete_distrib = Lattice(rng_seed=7)
stop_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=.005)
sol,data_obj = integrate(integrand, true_measure, discrete_distrib, stop_criterion)
print(data_obj)

# Sobol
integrand = QuickConstruct(
        dimension = 1,
        custom_fun = custom_fun)
true_measure = Lebesgue(dimension=1, lower_bound=-2, upper_bound=2)
discrete_distrib = Sobol(rng_seed=7)
stop_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=.005)
sol,data_obj = integrate(integrand, true_measure, discrete_distrib, stop_criterion)
print(data_obj)