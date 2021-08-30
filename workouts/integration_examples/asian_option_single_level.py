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
    abs_tol = .5):
    
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
    solution,data = CubBayesLatticeG(integrand,abs_tol=abs_tol,order=1,ptransform='Baker').integrate()
    print('%s%s'%(data,bar))

    # CubBayesNetG
    discrete_distrib = Sobol(dimension=dimension, graycode=False)
    integrand = AsianOption(discrete_distrib, volatility, start_price, strike_price, interest_rate, t_final, call_put, mean_type)
    solution,data = CubBayesNetG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

def asian_option_single_level_high_dimensions(abs_tol=.5):
    payoff = AsianOption(Sobol(52), start_price=30, strike_price=25)
    price, data = CubQMCSobolG(payoff, abs_tol=abs_tol).integrate()
    print(
        f'CubQMCSobolG            option price = ${price:.4f} using {data.time_integrate:.3f} seconds and {data.n_total:.2e} samples')

    payoff = AsianOption(Lattice(52, order='linear'), start_price=30, strike_price=25)
    price, data = CubBayesLatticeG(payoff, abs_tol=abs_tol, order=1, ptransform='Baker').integrate()
    print(
        f'CubBayesLatticeG        option price = ${price:.4f} using {data.time_integrate:.3f} seconds and {data.n_total:.2e} samples')

    payoff = AsianOption(Sobol(52), start_price=30, strike_price=25)
    price, data = CubBayesNetG(payoff, abs_tol=abs_tol).integrate()
    print(
        f'CubBayesNetG            option price = ${price:.4f} using {data.time_integrate:.3f} seconds and {data.n_total:.2e} samples')

    payoff = AsianOption(Sobol(52), start_price=100, strike_price=200)
    price, data = CubQMCSobolG(payoff, abs_tol=abs_tol).integrate()
    print(
        f'CubQMCSobolG            option price = ${price:.4f} using {data.time_integrate:.3f} seconds and {data.n_total:.2e} samples')

    payoff = AsianOption(BrownianMotion(Sobol(52), drift=1), start_price=100, strike_price=200)
    price, data = CubQMCSobolG(payoff, abs_tol=abs_tol).integrate()
    print(
        f'CubQMCSobolG with IS    option price = ${price:.4f} using {data.time_integrate:.3f} seconds and {data.n_total:.2e} samples')

    payoff = AsianOption(Lattice(52, order='linear'), start_price=100, strike_price=200)
    price, data = CubBayesLatticeG(payoff, abs_tol=abs_tol, order=1, ptransform='Baker').integrate()
    print(
        f'CubBayesLatticeG        option price = ${price:.4f} using {data.time_integrate:.3f} seconds and {data.n_total:.2e} samples')

    payoff = AsianOption(Sobol(52), start_price=100, strike_price=200)
    price, data = CubBayesNetG(payoff, abs_tol=abs_tol).integrate()
    print(
        f'CubBayesNetG            option price = ${price:.4f} using {data.time_integrate:.3f} seconds and {data.n_total:.2e} samples')

    print

if __name__ == "__main__":
    asian_option_single_level_high_dimensions(abs_tol=.025)
    asian_option_single_level(abs_tol=.025)