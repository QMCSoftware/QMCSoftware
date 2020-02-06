"""
Single level Asian Option
python workouts/integration_examples/asian_option_single_level.py  > outputs/integration_examples/asian_option_single_level.log
"""

from numpy import arange
from qmcpy import *

bar = '\n'+'~'*100+'\n'

def single_level_asian_option(
    time_vector = arange(1 / 64, 65 / 64, 1 / 64),
    volatility = .5,
    start_price = 30,
    strike_price = 25,
    interest_rate = .01,
    mean_type = 'geometric',
    abs_tol = .05):
    
    dimension = len(time_vector)
    print(bar)

    # CLT
    distribution = IIDStdGaussian(dimension, seed=7)
    measure = BrownianMotion(distribution,time_vector)
    integrand = AsianCall(measure, volatility, start_price, strike_price, interest_rate, mean_type)
    stopper = CLT(distribution,abs_tol=abs_tol)
    solution,data = integrate(stopper,integrand,measure,distribution)
    print('%s%s'%(data,bar))

    # CLTRep
    distribution = Lattice(dimension, scramble=True, replications=16, seed=7, backend='MPS')
    measure = BrownianMotion(distribution,time_vector)
    integrand = AsianCall(measure, volatility, start_price, strike_price, interest_rate, mean_type)
    stopper = CLTRep(distribution,abs_tol=abs_tol)
    solution,data = integrate(stopper,integrand,measure,distribution)
    print('%s%s'%(data,bar))

    # MeanMC_g
    distribution = IIDStdGaussian(dimension, seed=7)
    measure = BrownianMotion(distribution,time_vector)
    integrand = AsianCall(measure, volatility, start_price, strike_price, interest_rate, mean_type)
    stopper = MeanMC_g(distribution,abs_tol=abs_tol)
    solution,data = integrate(stopper,integrand,measure,distribution)
    print('%s%s'%(data,bar))

    # CubLattice
    distribution = Lattice(dimension=dimension, scramble=True, replications=0, seed=7, backend='GAIL')
    measure = BrownianMotion(distribution,time_vector)
    integrand = AsianCall(measure, volatility, start_price, strike_price, interest_rate, mean_type)
    stopper = CubLattice_g(distribution,abs_tol=abs_tol)
    solution,data = integrate(stopper,integrand,measure,distribution)
    print('%s%s'%(data,bar))


if __name__ == "__main__":
    
    single_level_asian_option()