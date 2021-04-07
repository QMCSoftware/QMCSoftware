import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin as fminsearch


def simple_lattice_gen(n, d, shift, firstBatch):
    # [xlat,xpts_un,xlat_un,xpts] =
    return

def keisterFunc(x, dim, a):
    return

def f_rand(xpts, rfun, a, b, c, seed):
    # fval
    return

def doPeriodTx(integrand, ptransform):
    return

def ObjectiveFunction(a, b, xlat, ftilde):
    return

#
# Minimum working example to test Gaussian diagnostics idea
#
def MWE_gaussian_diagnostics_engine(whEx, dim, npts, r, fpar, nReps, nPlots):
    # format short
    # close all
    # gail.InitializeDisplay

    # whEx = 3
    fNames = {'ExpCos', 'Keister', 'rand'}
    ptransforms = {'none', 'C1sin', 'none'}
    fName = fNames[whEx]
    ptransform = ptransforms[whEx]

    rOptAll = np.zeros((nReps, 1))
    thOptAll = np.zeros((nReps, 1))

    # parameters for random function
    # seed = 202326
    if whEx == 3:
        rfun = r / 2
        f_mean = fpar(3)
        f_std_a = fpar(1)  # this is square root of the a in the talk
        f_std_b = fpar(2)  # this is square root of the b in the talk
        theta = (f_std_a / f_std_b) ** 2
    else:
        theta = np.nan
    # end

    for iii in range(1, nReps):
        seed = seed = np.randint(low=1, high=1e6)  # randi([1, 1e6], 1, 1)  # different each rep
        shift = np.matlib.rand(1, dim)

        _, xlat, _, xpts = simple_lattice_gen(npts, dim, shift, True)

        if fName == 'ExpCos':
            integrand = lambda x: np.exp(sum(np.cos(2 * np.pi * x), 2))
        elif fName == 'Keister':
            integrand = lambda x: keisterFunc(x, dim, 1 / np.sqrt(2))  # a=0.8
        elif fName == 'rand':
            integrand = lambda x: f_rand(x, rfun, f_std_a, f_std_b, f_mean, seed)
        else:
            print('Invalid function name')
        # end

        integrand_p = doPeriodTx(integrand, ptransform)

        y = integrand_p(xpts)  # function data
        ftilde = np.fft(y)  # fourier coefficients
        ftilde[0] = 0  # ftilde = \mV**H(\vf - m \vone), subtract mean
        if dim == 1:
            hFigIntegrand = plt.figure()
            plt.scatter(xpts, y, 10)
            plt.title('%s_n-%d_Tx-%s'.format(fName, npts, ptransform))  #, 'interpreter', 'none')
            # saveas(hFigIntegrand, '%s_n-%d_Tx-%s_rFun-%1.2f.png'.format(fName, npts, ptransform, rfun))
        # end

    objfun = lambda lnParams: ObjectiveFunction(np.exp(lnParams[0]), 1 + np.exp(lnParams[1]), xlat, ftilde)
    ## Plot the objective function
    lnthetarange = np.range(-2, 2, 0.2)  # range of log(theta) for plotting
    lnorderrange = np.range(-1, 1, 0.1)  # range of log(r) for plotting
    [lnthth, lnordord] = np.meshgrid(lnthetarange, lnorderrange)
    objobj = lnthth
    for ii in range(0, lnthth.shape[0]):
        for jj in range(0, lnthth.shape[1]):
            objobj[ii, jj] = objfun([lnthth[ii, jj], lnordord[ii, jj]])
    # end
    # end
    if iii <= nPlots:
        figH = plt.figure()
        shandle = plt.surf(lnthth, lnordord, objobj)
        set(shandle, 'EdgeColor', 'none', 'facecolor', 'interp')
        set(gca, 'xtick', np.log([0.2 0.4 1 3 7]), 'xticklabel', {'0.2', '0.4', '1', '3', '7'}, ...
        'ytick', np.log([1.4 1.6 2 2.6 3.7] - 1), 'yticklabel', {'1.4', '1.6', '2', '2.6', '3.7'})
        plt.xlabel('\(\theta\)')
        plt.ylabel('\(r\)')
    # end

    [objMinAppx, which] = min(objobj, [], 'all', 'linear')
    [whichrow, whichcol] = ind2sub(lnthth.shape, which)
    lnthOptAppx = lnthth(whichrow, whichcol)
    thetaOptAppx = np.exp(lnthOptAppx)
    lnordOptAppx = lnordord(whichrow, whichcol)
    orderOptAppx = 1 + np.exp(lnordOptAppx)
    objMinAppx  # minimum objectiove function by brute force search

    ## Optimize the objective function
    [lnParamsOpt, objMin] = fminsearch(objfun, x1=lnthOptAppx, x2=lnordOptAppx, xtol=1e-3)
    print(objMin)  # minimum objectiove function by Nelder-Mead
    thetaOpt = np.exp(lnParamsOpt(1))
    rOpt = 1 + np.exp(lnParamsOpt(2))
    rOptAll[iii] = rOpt
    thOptAll[iii] = thetaOpt

    if iii <= nPlots:
        # hold on
        plt.scatter3(lnParamsOpt(1), lnParamsOpt(2), objfun(lnParamsOpt) * 1.002, 1000, MATLABYellow, '.')  # # end
        if np.isnan(theta):
            filename = f'{fName}-ObjFun-n-{npts}-d-{dim}-case-{iii}.jpg'
            plt.savefig(figH, filename)
        else:
            filename = f'{fName}-ObjFun-n-{npts}-d-{dim}-r-{r * 100}-th-{100 * theta}-case-{iii}.jpg'
            plt.savefig(figH, filename)
        # end
    # end

    # lambda1 = kernel(r, xlat_, thetaOpt)
    vlambda = kernel2(thetaOpt, rOpt, xlat)
    s2 = sum(abs(ftilde[2:] ** 2)/ vlambda[2:]) / (npts ** 2)
    vlambda = s2 * vlambda

    # apply transform
    # $\vZ = \frac 1n \mV \mLambda**{-\frac 12} \mV**H(\vf - m \vone)$
    # ifft also includes 1/n division
    vz = ifft(ftilde / np.sqrt(vlambda))
    vz_real = np.real(vz)  # vz must be real as intended by the transformation

    # create_plots('normplot')
    if iii <= nPlots:
        create_plots('qqplot', vz_real, fName, dim, iii, r, rOpt, theta, thetaOpt)
    # end
    print('r = %7.5f, rOpt = %7.5f, theta = %7.5f, thetaOpt = %7.5f\n'.format( r, rOpt, theta, thetaOpt)

    # end
    return [theta, rOptAll, thOptAll, fName]
# end

def save_plot_as_image(figH, filename):
    AxesH = gca  # Not the GCF
    F = getframe(AxesH)
    imwrite(F.cdata, filename)
    # end
