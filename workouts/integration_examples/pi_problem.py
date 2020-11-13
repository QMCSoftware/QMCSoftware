"""
Mathematica (Wolfram Alpha): integral_(-2)^2 sqrt(4 - x^2) (1./2 + x^3 cos(x/2)) dx = 3.14159
python workouts/integration_examples/free_wifi.py > outputs/integration_examples/free_wifi.log
"""

from qmcpy import *
from numpy import *
from time import time

def pi_problem(abs_tol=.01):
    t0 = time()
    distribution = Sobol(dimension=1, seed=7)
    measure = Lebesgue(distribution, lower_bound=-2, upper_bound=2)
    integrand = CustomFun(measure, lambda x: sqrt(4 - x**2) * (1. / 2 + x**3 * cos(x / 2)))
    solution,data = CubQMCSobolG(integrand, abs_tol=abs_tol, n_max=2**30).integrate()
    password = str(solution).replace('.', '')[:10]
    t_delta = time() - t0
    return password,t_delta,data

def pi_problem_bayes_net(abs_tol=.01):
    t0 = time()
    distribution = Sobol(dimension=1, seed=7, graycode=False)
    measure = Lebesgue(distribution, lower_bound=-2, upper_bound=2)
    integrand = CustomFun(measure, lambda x: sqrt(4 - x**2) * (1. / 2 + x**3 * cos(x / 2)))
    solution,data = CubBayesNetG(integrand, abs_tol=abs_tol, n_max=2**30).integrate()
    password = str(solution).replace('.', '')[:10]
    t_delta = time() - t0
    return password,t_delta,data

if __name__ == '__main__':
    print('CubBayesNetG:')
    password,total_time,data = pi_problem_bayes_net(abs_tol=4e-3) # give 3 significant figures of accuracy
    print("  Password:", password)
    print('  CPU time: %.2f sec'%total_time)  # very slow, takes much longer than CubQMCSobolG
    print('\n  '+'~'*100+'\n\n%s'%str(data))

    print('CubQMCSobolG:')
    password,total_time,data = pi_problem(abs_tol=4e-10) # give 10 significant figures of accuracy
    print("  Password:", password)  # 3141592653
    print('  CPU time: %.2f sec'%total_time)  # around 75 seconds
    print('\n  '+'~'*100+'\n\n%s'%str(data))

