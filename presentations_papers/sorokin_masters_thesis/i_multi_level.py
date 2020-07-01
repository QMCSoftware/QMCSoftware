from qmcpy import *
AsianCall(BrownianMotion(Lattice(dimension=8),drift=1),
    volatility=0.5, start_price=30, strike_price=35,
    interest_rate=0., mean_type='arithmetic',
    multi_level_dimensions=[4,8,16])
MLCallOptions(Gaussian(Sobol()),
    option='European', volatility=.2, start_strike_price=100,
    interest_rate=.05, t_final=1.)
MLCallOptions(Gaussian(Sobol()),
    option='Asian', volatility=.2, start_strike_price=100,
    interest_rate=.05, t_final=1.)