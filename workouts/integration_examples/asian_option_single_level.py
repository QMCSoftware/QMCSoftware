"""
Single level Asian Option
python workouts/integration_examples/asian_option_single_level.py  > outputs/integration_examples/asian_option_single_level.log
"""

from numpy import arange
from qmcpy import *

bar = '\n'+'~'*100+'\n'

def asian_option_single_level(
    volatility = .5,
    start_price = 30,
    strike_price = 25,
    interest_rate = .01,
    mean_type = 'geometric',
    abs_tol = .1):
    
    dimension = 64
    print(bar)

    # CLT
    distribution = IIDStdGaussian(dimension, seed=7)
    measure = BrownianMotion(distribution)
    integrand = AsianCall(measure, volatility, start_price, strike_price, interest_rate, mean_type)
    solution,data = CLT(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CLTRep
    distribution = Lattice(dimension, scramble=True, seed=7, backend='MPS')
    measure = BrownianMotion(distribution)
    integrand = AsianCall(measure, volatility, start_price, strike_price, interest_rate, mean_type)
    solution,data = CLTRep(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # MeanMC_g
    distribution = IIDStdGaussian(dimension, seed=7)
    measure = BrownianMotion(distribution)
    integrand = AsianCall(measure, volatility, start_price, strike_price, interest_rate, mean_type)
    solution,data = MeanMC_g(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubLattice_g
    distribution = Lattice(dimension=dimension, scramble=True, seed=7, backend='GAIL')
    measure = BrownianMotion(distribution)
    integrand = AsianCall(measure, volatility, start_price, strike_price, interest_rate, mean_type)
    solution,data = CubLattice_g(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubSobol_g
    '''
    distribution = Sobol(dimension=dimension, scramble=True, seed=7, backend='QRNG')
    measure = BrownianMotion(distribution)
    integrand = AsianCall(measure, volatility, start_price, strike_price, interest_rate, mean_type)
    solution,data = CubSobol_g(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))
    '''

if __name__ == "__main__":
    asian_option_single_level()