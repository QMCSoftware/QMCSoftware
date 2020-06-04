""" Comparing various stopping criterion on European Call option """

from qmcpy import *
from scipy.stats import norm

def tol_check(y,y_hat,tol,verbose=False):
    within_tol = abs(y-y_hat) < tol
    if not within_tol:
        raise Exception("Tolerence check failed\n\ty = %.4f\n\ty_hat = %.4f\n\ttol = %.4f"\
            %(y,y_hat,tol))
    if verbose:
        print('Within Tolerence:',within_tol)
    return

def european_options(abs_tol=.1):
    volatility = .2
    start_price = 100
    interest_rate = .05

    discrete_distribution = IIDStdGaussian()
    measure = Gaussian(discrete_distribution)
    integrand = MLMCCallOptions(measure,'european',volatility,start_price,interest_rate)
    algorithm = MLMC(integrand,abs_tol,n_init=256,n_max=1e10)
    sol0,data0 = algorithm.integrate()
    print('\n\nMLMC:\n%s\n%s'%('~'*100,data0),end='')
    exact_val = integrand.get_exact_value()
    tol_check(exact_val,sol0,abs_tol,verbose=True)

    measure = Gaussian(Lattice())
    integrand = MLMCCallOptions(measure,'european',volatility,start_price,interest_rate)
    algorithm = MLQMC(integrand,abs_tol,n_init=256,n_max=1e10,replications=16)
    sol1,data1 = algorithm.integrate()
    print('\n\nMLQMC:\n%s\n%s'%('~'*100,data1),end='')
    tol_check(exact_val,sol1,abs_tol,verbose=True)

    d = 2**(data1.levels-1) # use the same dimension as the finest level for MLQMC
    measure = BrownianMotion(Lattice(d))
    integrand = EuropeanOption(measure,volatility,start_price,start_price,interest_rate,call_put='call')
    algorithm = CubLattice_g(integrand,abs_tol)
    sol2,data2 = algorithm.integrate()
    print('\n\nCubLattice_g:\n%s\n%s'%('~'*100,data2),end='')
    tol_check(exact_val,sol2,abs_tol,verbose=True)

    measure = BrownianMotion(Lattice(d))
    integrand = EuropeanOption(measure,volatility,start_price,start_price,interest_rate,call_put='call')
    algorithm = CLTRep(integrand,abs_tol)
    sol3,data3 = algorithm.integrate()
    print('\n\nCLTRep:\n%s\n%s'%('~'*100,data3),end='')
    tol_check(exact_val,sol3,abs_tol,verbose=True)

if __name__ == '__main__':
    european_options(abs_tol=.01)