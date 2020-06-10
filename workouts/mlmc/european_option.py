""" Comparing various stopping criterion on European Call option """

from qmcpy import *
from scipy.stats import norm

tol_check = lambda y,y_hat,tol: print('Within Tolerence: %s'%(abs(y-y_hat) < tol))

def european_options(abs_tol=.1):
    volatility = .2
    start_price = 100
    interest_rate = .05

    discrete_distribution = IIDStdGaussian()
    measure = Gaussian(discrete_distribution)
    integrand = MLCallOptions(measure,'european',volatility,start_price,interest_rate)
    algorithm = CubMcMl(integrand,abs_tol,n_init=256,n_max=1e10)
    sol0,data0 = algorithm.integrate()
    print('\n\nMLMC:\n%s\n%s'%('~'*100,data0),end='')
    exact_val = integrand.get_exact_value()
    tol_check(exact_val,sol0,abs_tol)

    measure = Gaussian(Lattice())
    integrand = MLCallOptions(measure,'european',volatility,start_price,interest_rate)
    algorithm = CubQmcMl(integrand,abs_tol,n_init=256,n_max=1e10,replications=16)
    sol1,data1 = algorithm.integrate()
    print('\n\nMLQMC:\n%s\n%s'%('~'*100,data1),end='')
    tol_check(exact_val,sol1,abs_tol)

    d = 2**(data1.levels-1) # use the same dimension as the finest level for MLQMC
    measure = BrownianMotion(Lattice(d))
    integrand = EuropeanOption(measure,volatility,start_price,start_price,interest_rate,call_put='call')
    algorithm = CubQmcLatticeG(integrand,abs_tol)
    sol2,data2 = algorithm.integrate()
    print('\n\nCubLattice_g:\n%s\n%s'%('~'*100,data2),end='')
    tol_check(exact_val,sol2,abs_tol)

    measure = BrownianMotion(Lattice(d))
    integrand = EuropeanOption(measure,volatility,start_price,start_price,interest_rate,call_put='call')
    algorithm = CubQmcClt(integrand,abs_tol)
    sol3,data3 = algorithm.integrate()
    print('\n\nCLTRep:\n%s\n%s'%('~'*100,data3),end='')
    tol_check(exact_val,sol3,abs_tol)

if __name__ == '__main__':
    european_options(abs_tol=.025)