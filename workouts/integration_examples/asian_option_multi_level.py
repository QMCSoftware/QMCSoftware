"""
Multi level Asian Option
python workouts/integration_examples/asian_option_multi_level.py  > outputs/integration_examples/asian_option_multi_level.log
"""

from qmcpy import *

bar = '\n'+'~'*100+'\n'

def asian_option_multi_level(
    volatility = .5,
    start_price = 30,
    strike_price = 25,
    interest_rate = .01,
    mean_type = 'geometric',
    abs_tol = .5):
    
    print(bar)

    # CubMCCLT
    integrand = AsianOption(
        sampler = IIDStdUniform(seed=7),
        volatility = volatility,
        start_price = start_price,
        strike_price = strike_price,
        interest_rate = interest_rate,
        mean_type = mean_type,
        multilevel_dims = [4,16,64])
    solution,data = CubMCCLT(integrand, abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

if __name__ == "__main__":
    asian_option_multi_level(abs_tol=.025)