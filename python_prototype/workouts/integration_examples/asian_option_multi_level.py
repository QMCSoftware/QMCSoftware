"""
Multi level Asian Option
python workouts/integration_examples/asian_option_multi_level.py  > outputs/integration_examples/asian_option_multi_level.log
"""

from numpy import arange
from qmcpy import *

bar = '\n'+'~'*100+'\n'

def asian_option_multi_level(
    volatility = .5,
    start_price = 30,
    strike_price = 25,
    interest_rate = .01,
    mean_type = 'geometric',
    abs_tol = .1):
    
    levels = 3
    print(bar)

    # CLT
    distributions = MultiLevelConstructor(levels,
        IIDStdGaussian,
            dimension = [4,16,64],
            seed = arange(7,7+levels))
    measures = MultiLevelConstructor(levels,
        BrownianMotion,
            distribution = distributions)
    integrands = MultiLevelConstructor(levels,
        AsianCall,
            measure = measures,
            volatility = volatility,
            start_price = start_price,
            strike_price = strike_price,
            interest_rate = interest_rate,
            mean_type = mean_type)
    solution,data = CLT(integrands, abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))


if __name__ == "__main__":
    
    asian_option_multi_level()