""" Comparing various stopping criterion on European Call option """

from qmcpy import *
from scipy.stats import norm


def european_options(abs_tol=.1):
    volatility = .2
    start_price = 100
    interest_rate = .05
    t_final = 1

    integrand = MLCallOptions(IIDStdUniform(),'european',volatility,start_price,interest_rate,t_final)
    algorithm = CubMCML(integrand,abs_tol,n_init=256,n_max=1e10)
    sol0,data0 = algorithm.integrate()
    print('\n%s\n\n%s'%('~'*100,data0))
    exact_val = integrand.get_exact_value()

    integrand = MLCallOptions(Lattice(),'european',volatility,start_price,interest_rate,t_final)
    algorithm = CubQMCML(integrand,abs_tol,n_init=16,n_max=1e10,replications=16)
    sol1,data1 = algorithm.integrate()
    print('\n%s\n\n%s'%('~'*100,data1))

    d = 2**(data1.levels-1) # use the same dimension as the finest level for MLQMC
    integrand = EuropeanOption(Lattice(d),volatility,start_price,start_price,interest_rate,t_final,call_put='call')
    algorithm = CubQMCLatticeG(integrand,abs_tol)
    sol2,data2 = algorithm.integrate()
    print('\n%s\n\n%s'%('~'*100,data2))

    d = 2**(data1.levels-1) # use the same dimension as the finest level for MLQMC
    integrand = EuropeanOption(Sobol(d),volatility,start_price,start_price,interest_rate,t_final,call_put='call')
    algorithm = CubQMCSobolG(integrand,abs_tol)
    sol2,data2 = algorithm.integrate()
    print('\n%s\n\n%s'%('~'*100,data2))

    integrand = EuropeanOption(Lattice(d),volatility,start_price,start_price,interest_rate,t_final,call_put='call')
    algorithm = CubQMCCLT(integrand,abs_tol)
    sol3,data3 = algorithm.integrate()
    print('\n%s\n\n%s'%('~'*100,data3))

    # CubBayesLatticeG
    d = 2**(data1.levels-1) # use the same dimension as the finest level for MLQMC
    integrand = EuropeanOption(Lattice(d, order='linear'),volatility,start_price,start_price,interest_rate,t_final,call_put='call')
    algorithm = CubBayesLatticeG(integrand,abs_tol,ptransform='Baker')
    sol2,data2 = algorithm.integrate()
    print('\n%s\n\n%s'%('~'*100,data2))

    # CubBayesNetG
    d = 2**(data1.levels-1) # use the same dimension as the finest level for MLQMC
    integrand = EuropeanOption(Sobol(d),volatility,start_price,start_price,interest_rate,t_final,call_put='call')
    algorithm = CubBayesNetG(integrand,abs_tol)
    sol2,data2 = algorithm.integrate()
    print('\n%s\n\n%s'%('~'*100,data2))

if __name__ == '__main__':
    european_options(abs_tol=.025)