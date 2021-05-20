"""
Mathematica (Wolfram Alpha): integral_(-2)^2 sqrt(4 - x^2) (1./2 + x^3 cos(x/2)) dx = 3.14159
python workouts/integration_examples/free_wifi.py > outputs/integration_examples/free_wifi.log
"""

from qmcpy import *
from numpy import *
from time import time

def pi_problem(abs_tol=.5):
    t0 = time()
    d = 1
    integrand = CustomFun(
        true_measure = Lebesgue(Uniform(Sobol(d,seed=7), lower_bound=-2, upper_bound=2)), 
        g = lambda x: (sqrt(4 - x**2) * (1. / 2 + x**3 * cos(x / 2))).sum(1))
    solution,data = CubQMCSobolG(integrand, abs_tol=abs_tol, n_max=2**30).integrate()
    password = str(solution).replace('.', '')[:10]
    t_delta = time() - t0
    return password,t_delta,data

def pi_problem_bayes_lattice(abs_tol=.5):
    t0 = time()
    d = 1
    integrand = CustomFun(
        true_measure = Lebesgue(Uniform(Lattice(d, seed=7, order='linear'), lower_bound=-2, upper_bound=2)), 
        g = lambda x: (sqrt(4 - x**2) * (1. / 2 + x**3 * cos(x / 2))).sum(1))
    solution,data = CubBayesLatticeG(integrand, abs_tol=abs_tol, n_max=2**30).integrate()
    password = str(solution).replace('.', '')[:10]
    t_delta = time() - t0
    return password,t_delta,data

def pi_problem_bayes_net(abs_tol=.5):
    t0 = time()
    d = 1
    integrand = CustomFun(
        true_measure = Lebesgue(Uniform(Sobol(d, seed=7), lower_bound=-2, upper_bound=2)),
        g = lambda x: (sqrt(4 - x**2) * (1. / 2 + x**3 * cos(x / 2))).sum(1))
    solution,data = CubBayesNetG(integrand, abs_tol=abs_tol, n_max=2**30).integrate()
    password = str(solution).replace('.', '')[:10]
    t_delta = time() - t0
    return password,t_delta,data

if __name__ == '__main__':
    print('CubBayesLatticeG:')
    password, total_time, data = pi_problem_bayes_lattice(abs_tol=4e-10)  # give 3 significant figures of accuracy
    print("  Password:", password)
    print('  CPU time: %.2f sec' % total_time)  # around 12 seconds
    print('\n  ' + '~' * 100 + '\n\n%s' % str(data))

    '''
    print('\nCubBayesNetG:')
    password,total_time,data = pi_problem_bayes_net(abs_tol=4e-8) # give 3 significant figures of accuracy
    print("  Password:", password)
    print('  CPU time: %.2f sec'%total_time)  # slow, takes much longer than CubQMCSobolG
    print('\n  '+'~'*100+'\n\n%s'%str(data))  # CPU time: 230.69 sec, n_total 8388608, Password: 3141592653

    print('\nCubQMCSobolG:')
    password,total_time,data = pi_problem(abs_tol=4e-10) # give 10 significant figures of accuracy
    print("  Password:", password)  # 3141592653
    print('  CPU time: %.2f sec'%total_time)  # around 75 seconds
    print('\n  '+'~'*100+'\n\n%s'%str(data))
'''