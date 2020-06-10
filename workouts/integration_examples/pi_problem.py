"""
Mathematica (Wolfram Alpha): integral_(-2)^2 sqrt(4 - x^2) (1/2 + x^3 cos(x/2)) dx = 3.14159
python workouts/integration_examples/free_wifi.py > outputs/integration_examples/free_wifi.log
"""

from qmcpy import *
from numpy import *
from time import perf_counter

def pi_problem(abs_tol=.01):
    t0 = perf_counter()
    distribution = Sobol(dimension=1, seed=7)
    measure = Lebesgue(distribution, lower_bound=-2, upper_bound=2)
    integrand = CustomFun(measure, lambda x: sqrt(4 - x**2) * (1 / 2 + x**3 * cos(x / 2)))
    solution,data = CubQmcSobolG(integrand, abs_tol=abs_tol, n_max=2**30).integrate()
    password = str(solution).replace('.', '')[:10]
    time = perf_counter() - t0
    return password,time,data  

if __name__ == '__main__':
    password,time,data = pi_problem(abs_tol=4e-10) # give 10 significant figures of accuracy
    print("Password:", password)  # 3141592653
    print('CPU time: %.2f sec'%time)  # around 75 seconds
    print('\n'+'~'*100+'\n\n%s'%str(data))