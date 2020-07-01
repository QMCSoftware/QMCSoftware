from qmcpy import *
CustomFun(Uniform(Sobol(dimension=2)),
    custom_fun = lambda x: x[:,0]*x[:,1])
Linear(Gaussian(Sobol(dimension=2)))
Keister(Gaussian(Sobol(dimension=2),covariance=1./2))
EuropeanOption(BrownianMotion(Sobol(dimension=16),drift=1),
    volatility=0.5, start_price=30, strike_price=35,
    interest_rate=0., call_put='call')
AsianCall(BrownianMotion(Lattice(dimension=8),drift=1),
    volatility=0.5, start_price=30, strike_price=35,
    interest_rate=0., mean_type='arithmetic')
