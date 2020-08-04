"""
Single level Asian Option
python workouts/integration_examples/asian_option_single_level.py  > outputs/integration_examples/asian_option_single_level.log
"""

from qmcpy import *

bar = '\n'+'~'*100+'\n'

def asian_option_single_level(
    volatility = .5,
    start_price = 30,
    strike_price = 25,
    interest_rate = .01,
    call_put = 'call',
    mean_type = 'geometric',
    abs_tol = .1):
    
    dimension = 64
    print(bar)

    # CubMCCLT
    distribution = IIDStdGaussian(dimension, seed=7)
    measure = BrownianMotion(distribution)
    integrand = AsianOption(measure, volatility, start_price, strike_price, interest_rate, call_put, mean_type)
    solution,data = CubMCCLT(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubQMCCLT
    distribution = Lattice(dimension, randomize=True, seed=7, backend='MPS')
    measure = BrownianMotion(distribution)
    integrand = AsianOption(measure, volatility, start_price, strike_price, interest_rate, call_put, mean_type)
    solution,data = CubQMCCLT(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubMCG
    distribution = IIDStdGaussian(dimension, seed=7)
    measure = BrownianMotion(distribution)
    integrand = AsianOption(measure, volatility, start_price, strike_price, interest_rate, call_put, mean_type)
    solution,data = CubMCG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubQMCLatticeG
    distribution = Lattice(dimension=dimension, randomize=True, seed=7, backend='GAIL')
    measure = BrownianMotion(distribution)
    integrand = AsianOption(measure, volatility, start_price, strike_price, interest_rate, call_put, mean_type)
    solution,data = CubQMCLatticeG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubQMCSobolG
    distribution = Sobol(dimension=dimension, randomize=True, seed=7, backend='QRNG')
    measure = BrownianMotion(distribution)
    integrand = AsianOption(measure, volatility, start_price, strike_price, interest_rate, call_put, mean_type)
    solution,data = CubQMCSobolG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubBayesLatticeG
    distribution = Lattice(dimension=dimension, randomize=False, linear=True, backend='GAIL')
    measure = BrownianMotion(distribution)
    integrand = AsianOption(measure, volatility, start_price, strike_price, interest_rate, call_put, mean_type)
    solution,data = CubBayesLatticeG(integrand,abs_tol=abs_tol,ptransform='Baker').integrate()
    print('%s%s'%(data,bar))

if __name__ == "__main__":
    asian_option_single_level(abs_tol=.025)