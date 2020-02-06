"""
Mathematica (Wolfram Alpha): integral_(-2)^2 sqrt(4 - x^2) (1/2 + x^3 cos(x/2)) dx = 3.14159
python workouts/integration_examples/free_wifi.py > outputs/integration_examples/free_wifi.log
"""

from qmcpy import *
from numpy import *
from time import process_time

t0 = process_time()
distribution = Lattice(dimension=1, replications=16, scramble=True, seed=7, backend='MPS')
measure = Lebesgue(distribution, lower_bound=-2, upper_bound=2)
integrand = QuickConstruct(measure, lambda x: sqrt(4 - x**2) * (1 / 2 + x**3 * cos(x / 2)))
stoppper = CLTRep(distribution, abs_tol=2.5e-10, n_max=2**30)
solution,data = integrate(stoppper, integrand, measure, distribution)
password = str(solution).replace('.', '')[:10]
print("Password:", password)  # 3141592653
print('CPU time: %.2f sec' % (process_time() - t0))  # around 30 seconds
print('\n'+'~'*100+'\n\n%s'%str(data))