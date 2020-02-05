from numpy import *
from qmcpy import *

time_vector = arange(1 / 64, 65 / 64, 1 / 64)
distribution = Lattice(dimension=len(time_vector), scramble=True, replications=16, seed=7, backend='MPS')
measure = BrownianMotion(distribution, time_vector=time_vector)
integrand = AsianCall(
    measure = measure,
    volatility = 0.5,
    start_price = 30,
    strike_price = 25,
    interest_rate = 0.01,
    mean_type = 'arithmetic')
stopper = CLTRep(distribution, abs_tol=.05)
solution,data = integrate(stopper, integrand, measure, distribution)
print(data)