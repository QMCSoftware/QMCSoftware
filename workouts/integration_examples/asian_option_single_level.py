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
    t_final = 1,
    call_put = 'call',
    mean_type = 'geometric',
    abs_tol = .1):
    
    dimension = 64
    print(bar)

    # CubMCCLT
    discrete_distrib = IIDStdUniform(dimension, seed=7)
    integrand = AsianOption(discrete_distrib, volatility, start_price, strike_price, interest_rate, t_final, call_put, mean_type)
    solution,data = CubMCCLT(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubQMCCLT
    discrete_distrib = Lattice(dimension, randomize=True, seed=7, order='MPS')
    integrand = AsianOption(discrete_distrib, volatility, start_price, strike_price, interest_rate, t_final, call_put, mean_type)
    solution,data = CubQMCCLT(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubMCG
    discrete_distrib = IIDStdUniform(dimension, seed=7)
    integrand = AsianOption(discrete_distrib, volatility, start_price, strike_price, interest_rate, t_final, call_put, mean_type)
    solution,data = CubMCG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubQMCLatticeG
    discrete_distrib = Lattice(dimension=dimension, randomize=True, seed=7)
    integrand = AsianOption(discrete_distrib, volatility, start_price, strike_price, interest_rate, t_final, call_put, mean_type)
    solution,data = CubQMCLatticeG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubQMCSobolG
    discrete_distrib = Sobol(dimension=dimension, randomize=True, seed=7)
    integrand = AsianOption(discrete_distrib, volatility, start_price, strike_price, interest_rate, t_final, call_put, mean_type)
    solution,data = CubQMCSobolG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubBayesLatticeG
    discrete_distrib = Lattice(dimension=dimension, order='linear', randomize=True)
    integrand = AsianOption(discrete_distrib, volatility, start_price, strike_price, interest_rate, t_final, call_put, mean_type)
    solution,data = CubBayesLatticeG(integrand,abs_tol=abs_tol,ptransform='Baker').integrate()
    print('%s%s'%(data,bar))

    # CubBayesNetG
    discrete_distrib = Sobol(dimension=dimension, randomize='LMS', graycode=False)
    integrand = AsianOption(discrete_distrib, volatility, start_price, strike_price, interest_rate, t_final, call_put, mean_type)
    solution,data = CubBayesNetG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

if __name__ == "__main__":
    asian_option_single_level(abs_tol=.025)