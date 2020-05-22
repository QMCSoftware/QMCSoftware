"""
Mathematica (Wolfram Alpha): integral_(-2)^2 sqrt(4 - x^2) (1/2 + x^3 cos(x/2)) dx = 3.14159
python workouts/integration_examples/free_wifi.py > outputs/integration_examples/free_wifi.log
"""

from qmcpy import *
from numpy import *
from time import perf_counter

t0 = perf_counter()
abs_tol = 1e-9
distribution = Sobol(dimension=1, seed=7, backend='QRNG')
measure = Lebesgue(distribution, lower_bound=-2, upper_bound=2)
integrand = QuickConstruct(measure, lambda x: sqrt(4 - x**2) * (1 / 2 + x**3 * cos(x / 2)))
solution,data = CubSobol_g(integrand, abs_tol=abs_tol, n_max=2**30).integrate()
password = str(solution).replace('.', '')[:10]
print("Password:", password)  # 3141592653
print('CPU time: %.2f sec' % (perf_counter() - t0))  # around 30 seconds
print('\n'+'~'*100+'\n\n%s'%str(data))