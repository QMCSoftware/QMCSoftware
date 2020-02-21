"""
Multi level Asian Option
python workouts/integration_examples/asian_option_multi_level.py  > outputs/integration_examples/asian_option_multi_level.log
"""

from numpy import arange
from qmcpy import *

bar = '\n'+'~'*100+'\n'

def asian_option_multi_level(
    time_vector = [
        arange(1/4, 5/4, 1/4),
        arange(1 / 64, 65 / 64, 1 / 64),
        arange(1/64, 65/64, 1/64)],
    volatility = .5,
    start_price = 30,
    strike_price = 25,
    interest_rate = .01,
    mean_type = 'geometric',
    abs_tol = .1):
    
    levels = len(time_vector)
    print(bar)

    # CLT
    distributions = MultiLevelConstructor(levels,
        IIDStdGaussian,
            dimension = [len(tv) for tv in time_vector],
            seed = 7)
    measures = MultiLevelConstructor(levels,
        BrownianMotion,
            distribution = distributions,
            time_vector = time_vector)
    integrands = MultiLevelConstructor(levels,
        AsianCall,
            measure = measures,
            volatility = volatility,
            start_price = start_price,
            strike_price = strike_price,
            interest_rate = interest_rate,
            mean_type = mean_type)
    algorithm = CLT(integrands, abs_tol=abs_tol)
    solution,data = algorithm.integrate()
    print('%s%s'%(data,bar))


if __name__ == "__main__":
    
    asian_option_multi_level()