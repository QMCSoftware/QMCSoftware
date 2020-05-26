""" Test various call options with MLMC StoppingCriterion """

from qmcpy import IIDStdGaussian, Gaussian, CallOptions
from workouts.mlmc.mlmc_test import mlmc_test

def mcqmc06(l_convergence=3,epsilons=[.05,.1]):
    for option in ['European','Asian']:
        print('\n\n'+'~'*100+'\n\n')
        print('%s Call Option'%option)
        distribution = IIDStdGaussian()
        measure = Gaussian(distribution)
        integrand = CallOptions(measure,
            option = option,
            volatility = .2,
            start_strike_price = 100, 
            interest_rate = .05,
            t_final = 1)
        mlmc_test(integrand,
            n = 20000, # samples for convergence tests
            l = l_convergence, # levels for convergence tests 
            n0 = 200,
            eps = epsilons,
            l_min = 2,
            l_max = 10)
        print('\nExact Value: %s'%integrand.get_exact_value())
        
if __name__ == '__main__':
    mcqmc06(l_convergence=8,epsilons=[.005, 0.01, 0.02, 0.05, 0.1])