from numpy import *
from qmcpy import *

time_vector = [
    arange(1/4,5/4,1/4),
    arange(1/16,17/16,1/16),
    arange(1/64,65/64,1/64)]
levels = len(time_vector)
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
        volatility = 0.5,
        start_price = 30,
        strike_price = 25,
        interest_rate = 0.1,
        mean_type = 'arithmetic')
stopper = CLT(distributions, abs_tol=.05)
solution,data = integrate(stopper, integrands, measures, distributions)
print(data)