"""
Sample Measure objects and usage
python workouts/example_constructions/measure.py > outputs/example_constructions/measures.log
"""

from qmcpy import *
from copy import deepcopy
from numpy import *
set_printoptions(threshold=1e10)

d = 2
r = 0
seed = 7
iid_std_u = IIDStdUniform(dimension=d, seed=seed)
iid_std_g = IIDStdGaussian(dimension=d, seed=seed)
n = 8

bar = '\n'+'~'*100+'\n'
print(bar)

# Brownian motion
time_vector = [.5,1]
for distrib in [iid_std_u,iid_std_g]:
    measure = BrownianMotion(distribution=distrib, time_vector=time_vector)
    x = measure.gen_samples(n=n)
    print(measure)
    print(bar)

# Gaussian
mu = [3, 3]
sigma = [5, 7]
for distrib in [iid_std_u,iid_std_g]:
    measure = Gaussian(distribution=distrib, mean=mu, variance=sigma)
    x = measure.gen_samples(n=n)
    print(measure)
    print(bar)

# Uniform
a = [-2, -3]
b = 4
for distrib in [iid_std_u,iid_std_g]:
    measure = Uniform(distribution=distrib, lower_bound=a, upper_bound=b)
    x = measure.gen_samples(n=n)
    print(measure)
    print(bar)

# Lebesgue
a = [-2, -3]
b = [2, 4]
for distrib in [iid_std_u]:
    measure = Lebesgue(distribution=distrib, lower_bound=a, upper_bound=b)
    x = measure.gen_samples(n=n)
    print(measure)
    print(bar)
