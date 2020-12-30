""" Test mlml """

from qmcpy import CubMCML
from numpy import *
from numpy.linalg import lstsq

def mlmc_test(integrand_qmcpy, n, l, n0, eps, l_min, l_max):
    """
    Multilevel Monte Carlo test routine

    Args:
        integrand_qmcpy (function):
            low-level routine for l level estimation such that 
                Args:
                    x (ndarray): nx(integrand._dim_at_level(l)) array of samples from discrete distribution
                    l (int): level
                Return:    
                    sums(1) = sum(Pf-Pc)
                    sums(2) = sum((Pf-Pc).^2)
                    sums(3) = sum((Pf-Pc).^3)
                    sums(4) = sum((Pf-Pc).^4)
                    sums(5) = sum(Pf)
                    sums(6) = sum(Pf.^2)
                    cost = user-defined computational cost
        n (int): number of samples for convergence tests
        l (int): number of levels for convergence tests
        n0 (int): initial number of samples for MLMC calcs
        eps (float): desired accuracy array for MLMC calcs
        l_min (int): minimum number of levels for MLMC calcs
        l_max (int): maximum number of levels for MLMC calcs
    """
    # first, convergence tests
    n = 100*ceil(n/100) # make N a multiple of 100
    print('\nConvergence tests, kurtosis, telescoping sum check using N =%7d samples'%n)
    print('\t%-15s%-15s%-15s%-15s%-15s%-15s%-15s%s'\
        %('l','ave(Pf-Pc)','ave(Pf)','var(Pf-Pc)','var(Pf)','kurtosis','check','cost'))
    del1 = array([])
    del2 = array([])
    var1 = array([])
    var2 = array([])
    kur1 = array([])
    chk1 = array([])
    cost = array([])
    for ll in range(l+1):
        sums = 0
        cst = 0
        for j in range(1,101):
            # reset dimension
            new_dim = integrand_qmcpy._dim_at_level(ll)
            integrand_qmcpy.discrete_distrib.set_dimension(new_dim)
            integrand_qmcpy.true_measure.set_dimension(new_dim)
            # evaluate integral at sampleing points samples
            samples = integrand_qmcpy.discrete_distrib.gen_samples(n=n/100)
            integrand_qmcpy.f(samples,l=ll)
            sums_j = integrand_qmcpy.sums
            cst_j = integrand_qmcpy.cost
            sums = sums + sums_j/n
            cst = cst + cst_j/n
        if ll == 0:
            kurt = 0.
        else:
            kurt = ( sums[3] - 4*sums[2]*sums[0] + 6*sums[1]*sums[0]**2 - 
                     3*sums[0]*sums[0]**3 ) /  (sums[1]-sums[0]**2)**2.
        cost = hstack((cost, cst))
        del1 = hstack((del1, sums[0]))
        del2 = hstack((del2, sums[4]))
        var1 = hstack((var1, sums[1]-sums[0]**2))
        var2 = hstack((var2, sums[5]-sums[4]**2))
        var2 = maximum(var2, 1e-10) # fix for cases with var=0
        kur1 = hstack((kur1, kurt))
        if ll == 0:
            check = 0
        else:
            check = abs( del1[ll] + del2[ll-1] - del2[ll]) / \
                    ( 3.*( sqrt(var1[ll]) + sqrt(var2[ll-1]) + sqrt(var2[ll]) ) / sqrt(n))
        chk1 = hstack((chk1, check))

        print('\t%-15d%-15.4e%-15.4e%-15.3e%-15.3e%-15.2e%-15.2e%.2e'\
              %(ll,del1[ll],del2[ll],var1[ll],var2[ll],kur1[ll],chk1[ll],cst))
    # print out a warning if kurtosis or consistency check looks bad
    if kur1[-1] > 100.:
        print('WARNING: kurtosis on finest level = %f'%kur1[-1])
        print(' indicates MLMC correction dominated by a few rare paths;')
        print(' for information on the connection to variance of sample variances,')
        print(' see http://mathworld.wolfram.com/SampleVarianceDistribution.html\n')
    if max(chk1) > 1.:
        print('WARNING: maximum consistency error = %f'%max(chk1))
        print(' indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied;')
        print(' to be more certain, re-run mlmc_test with larger N\n')
    # use linear regression to estimate alpha, beta and gamma
    l1 = 2
    l2 = l+1
    x = ones((l2+1-l1,2))
    x[:,1] = arange(l1,l2+1)
    pa = lstsq(x,log2(absolute(del1[(l1-1):l2])),rcond=None)[0]
    alpha = -pa[1]
    pb = lstsq(x,log2(absolute(var1[(l1-1):l2])),rcond=None)[0]
    beta = -pb[1]
    pg = lstsq(x,log2(absolute(cost[(l1-1):l2])),rcond=None)[0]
    gamma = pg[1]
    print('\nLinear regression estimates of MLMC parameters')
    print('\talpha = %f  (exponent for MLMC weak convergence)'%alpha)
    print('\tbeta  = %f  (exponent for MLMC variance)'%beta)
    print('\tgamma = %f  (exponent for MLMC cost)'%gamma)
    #second, mlmc complexity tests
    print('\nMLMC complexity tests')
    print('\t%-15s%-15s%-15s%-15s%-15s%s'\
        %('eps','value','mlmc_cost','std_cost','savings','N_l'))
    alpha = max(alpha,0.5)
    beta  = max(beta,0.5)
    theta = 0.25
    for i in range(len(eps)):
        mlmc_qmcpy = CubMCML(integrand_qmcpy,
            rmse_tol = eps[i],
            n_init = n0,
            levels_min = l_min,
            levels_max = l_max,
            alpha0 = alpha,
            beta0 = beta,
            gamma0 = gamma)
        mlmc_qmcpy.integrate()
        p = mlmc_qmcpy.data.solution
        nl = mlmc_qmcpy.data.n_level
        cl = mlmc_qmcpy.data.cost_per_sample
        mlmc_cost = sum(nl*cl)
        idx = min(len(var2),len(nl))-1
        std_cost = var2[idx]*cl[-1] / ((1.-theta)*array(eps[i])**2)
        print('\t%-15.3e%-15.3e%-15.3e%-15.3e%-15.2f%s'\
            %(eps[i], p, mlmc_cost, std_cost, std_cost/mlmc_cost,'\t'.join(str(int(nli)) for nli in nl)))
