import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin as fminsearch


def simple_lattice_gen(n, d, shift, firstBatch):
    return  # [xlat,xpts_un,xlat_un,xpts]


def keisterFunc(x, dim, a):
    return


def f_rand(xpts, rfun, a, b, c, seed):
    return  # fval


def doPeriodTx(integrand, ptransform):
    return


def ObjectiveFunction(a, b, xlat, ftilde):
    return


def kernel2(thetaOpt, rOpt, xlat):
    return  # vlambda


def create_plots(type, vz_real, fName, dim, iii, r, rOpt, theta, thetaOpt):
    hFigNormplot, axFigNormplot = plt.subplots(subplot_kw={"projection": "3d"})

    plt.rcParams.update({'font.size': 12})

    # set(hFigNormplot,'defaultaxesfontsize',16,
    #   'defaulttextfontsize',12,   # make font larger
    #   'defaultLineLineWidth',2, 'defaultLineMarkerSize',6)
    n = len(vz_real)
    if type == 'normplot':
        axFigNormplot.normplot(vz_real)
    else:
        # ((1:n)-1/2)'/n
        q = np.arange(1, n) - 1 / 2
        stNorm = np.norminv(q)  # quantiles of standard normal
        axFigNormplot.plot(stNorm, sorted(vz_real), '.', 'MarkerSize', 20)
        # hold on:
        axFigNormplot.plot([-3, 3], [-3, 3], '-', 'linewidth', 4)
        axFigNormplot.set_xlabel('Standard Gaussian Quantiles')
        axFigNormplot.set_ylabel('Data Quantiles')

    if np.isnan(theta):
        plt_title = f'$d={dim}, n={n}, r_{{opt}}={rOpt:1.2f} , \\theta_{{opt}}={thetaOpt:1.2f} '
    else:
        plt_title = f'$d={dim}, n={n}, r={r:1.3f}, r_{{opt}}={rOpt:1.3f}, \\theta={theta:1.3f}, \\theta_{{opt}}={thetaOpt:1.3f} '

    axFigNormplot.set_title(plt_title)
    if np.isnan(theta):
        plt_filename = f'{fName}-QQPlot-n-{n}-d-{dim}-{iii}.jpg'
    else:
        plt_filename = f'{fName}-QQPlot-n-{n}-d-{dim}-r-{r * 100}-th-{100 * theta}-{iii}.jpg'
    plt.savefig(plt_filename)


#
# Minimum working example to test Gaussian diagnostics idea
#
def MWE_gaussian_diagnostics_engine(whEx, dim, npts, r, fpar, nReps, nPlots):
    # format short
    # close all
    # gail.InitializeDisplay

    # whEx = 3
    fNames = ['ExpCos', 'Keister', 'rand']
    ptransforms = ['none', 'C1sin', 'none']
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

    for iii in range(nReps):
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
            return

        integrand_p = doPeriodTx(integrand, ptransform)

        y = integrand_p(xpts)  # function data
        ftilde = np.fft.fft(y)  # fourier coefficients
        ftilde[0] = 0  # ftilde = \mV**H(\vf - m \vone), subtract mean
        if dim == 1:
            hFigIntegrand = plt.figure()
            plt.scatter(xpts, y, 10)
            plt.title('%s_n-%d_Tx-%s'.format(fName, npts, ptransform))  # , 'interpreter', 'none')
            # saveas(hFigIntegrand, '%s_n-%d_Tx-%s_rFun-%1.2f.png'.format(fName, npts, ptransform, rfun))

        objfun = lambda lnParams: ObjectiveFunction(np.exp(lnParams[0]), 1 + np.exp(lnParams[1]), xlat, ftilde)
        ## Plot the objective function
        lnthetarange = np.range(-2, 2, 0.2)  # range of log(theta) for plotting
        lnorderrange = np.range(-1, 1, 0.1)  # range of log(r) for plotting
        [lnthth, lnordord] = np.meshgrid(lnthetarange, lnorderrange)
        objobj = lnthth
        for ii in range(0, lnthth.shape[0]):
            for jj in range(0, lnthth.shape[1]):
                objobj[ii, jj] = objfun([lnthth[ii, jj], lnordord[ii, jj]])

        figH, axH = None, None
        if iii <= nPlots:
            from matplotlib import cm

            figH, axH = plt.subplots(subplot_kw={"projection": "3d"})

            shandle = axH.plot_surface(lnthth, lnordord, objobj, cmap=cm.coolwarm,
                                       linewidth=0, antialiased=False)
            axH.set_xticks(np.log([0.2, 0.4, 1, 3, 7]))
            axH.set_yticks(np.log([1.4, 1.6, 2, 2.6, 3.7] - 1))
            axH.xaxis.set_major_formatter('{x:.01f}')
            axH.yaxis.set_major_formatter('{x:.01f}')

            # set(shandle, 'EdgeColor', 'none', 'facecolor', 'interp')
            # set(gca, 'xtick', np.log([0.2, 0.4, 1, 3, 7]), 'xticklabel', {'0.2', '0.4', '1', '3', '7'}, ...
            #   'ytick', np.log([1.4, 1.6, 2, 2.6, 3.7] - 1), 'yticklabel', {'1.4', '1.6', '2', '2.6', '3.7'})
            axH.set_xlabel('\(\theta\)')
            axH.set_ylabel('\(r\)')

        [objMinAppx, which] = min(objobj, [], 'all', 'linear')
        # [whichrow, whichcol] = ind2sub(lnthth.shape, which)
        [whichrow, whichcol] = np.unravel_index(lnthth.shape, which)
        lnthOptAppx = lnthth[whichrow, whichcol]
        thetaOptAppx = np.exp(lnthOptAppx)
        lnordOptAppx = lnordord[whichrow, whichcol]
        orderOptAppx = 1 + np.exp(lnordOptAppx)
        print(objMinAppx)  # minimum objectiove function by brute force search

        ## Optimize the objective function
        [lnParamsOpt, objMin] = fminsearch(objfun, x1=lnthOptAppx, x2=lnordOptAppx, xtol=1e-3)
        print(objMin)  # minimum objective function by Nelder-Mead
        thetaOpt = np.exp(lnParamsOpt[0])
        rOpt = 1 + np.exp(lnParamsOpt[1])
        rOptAll[iii] = rOpt
        thOptAll[iii] = thetaOpt

        if iii <= nPlots:
            # hold on

            plt.scatter(lnParamsOpt[0], lnParamsOpt[1], objfun(lnParamsOpt) * 1.002, 1000, color='MATLABYellow',
                        marker='.')
            if np.isnan(theta):
                filename = f'{fName}-ObjFun-n-{npts}-d-{dim}-case-{iii}.jpg'
                plt.savefig(figH, filename)
            else:
                filename = f'{fName}-ObjFun-n-{npts}-d-{dim}-r-{r * 100}-th-{100 * theta}-case-{iii}.jpg'
                plt.savefig(figH, filename)

        # lambda1 = kernel(r, xlat_, thetaOpt)
        vlambda = kernel2(thetaOpt, rOpt, xlat)
        s2 = sum(abs(ftilde[2:] ** 2) / vlambda[2:]) / (npts ** 2)
        vlambda = s2 * vlambda

        # apply transform
        # $\vZ = \frac 1n \mV \mLambda**{-\frac 12} \mV**H(\vf - m \vone)$
        # ifft also includes 1/n division
        vz = np.fft.fft(ftilde / np.sqrt(vlambda))
        vz_real = np.real(vz)  # vz must be real as intended by the transformation

        # create_plots('normplot')
        if iii <= nPlots:
            create_plots('qqplot', vz_real, fName, dim, iii, r, rOpt, theta, thetaOpt)

        print('r = %7.5f, rOpt = %7.5f, theta = %7.5f, thetaOpt = %7.5f\n'.format(r, rOpt, theta, thetaOpt))

    return [theta, rOptAll, thOptAll, fName]


def save_plot_as_image(figH, filename):
    # AxesH = gca  # Not the GCF
    # F = getframe(AxesH)
    # imwrite(F.cdata, filename)
    figH.savefig(filename)


# gaussian random function
def f_rand(xpts, rfun, a, b, c, seed):
    dim = xpts.shape[1]
    np.random.seed(seed)  # initialize random number generator for reproducability
    N1 = 2 ** np.floor(16 / dim)
    Nall = N1 ** dim
    kvec = np.zeros([dim, Nall])  # initialize kved
    kvec[0, 0:N1 - 1] = range(0, N1 - 1)  # first dimension
    Nd = N1
    for d in range(2, dim):
        Ndm1 = Nd
        Nd = Nd * N1
        kvec[1:d, 1:Nd] = np.vstack(np.repmat(kvec[1:d - 1, 1:Ndm1], 1, N1),
                                    np.reshape(np.repmat(np.range(0, N1 - 1), Ndm1, 1), 1, Nd))

    f_c = a * np.random.randn(1, Nall - 1)
    f_s = a * np.random.randn(1, Nall - 1)
    f_0 = c + b * np.random.randn(1, 1)
    kbar = np.prod(max(kvec[:, 2:Nall], 1), 1)
    argx = (2 * np.pi * xpts) * kvec[:, 2:Nall]
    f_c_ = (f_c / kbar ** (rfun)) * np.cos(argx)
    f_s_ = (f_s / kbar ** (rfun)) * np.sin(argx)
    fval = f_0 + sum(f_c_ + f_s_, 2)
    # figure plot(fval, '.')
    # end
    return fval


def guassian_diagnostics_test():
    pass


if __name__ == '__main__':
    # gail.InitializeWorkspaceDisplay

    ## Exponential Cosine example
    fwh = 1
    dim = 3
    npts = 2 ** 6
    nRep = 20
    nPlot = 2
    [_, rOptAll, thOptAll, fName] = \
        MWE_gaussian_diagnostics_engine(fwh, dim, npts, [], [], nRep, nPlot)

    ## Plot Exponential Cosine example
    figH = plt.figure()
    plt.plot(rOptAll, thOptAll, marker='.', markersize=20, color='blue')
    # axis([4 6 0.1 10])
    # set(gca,'yscale','log')
    plt.title(f'\(d = {dim},\ n = {npts}\)')
    plt.xlabel('Inferred \(r\)')
    plt.ylabel('Inferred \(\theta\)')
    # print('-depsc',[fName '-rthInfer-n-' int2str(npts) '-d-' \
    #   int2str(dim)])
    figH.savefig(f'{fName}-rthInfer-n-{npts}-d-{dim}.jpg')

    ## Tests with random function
    rArray = [1.5, 2, 4]
    nrArr = len(rArray)
    fParArray = [0.5, 1, 2, 1, 1, 1, 1, 1, 1]
    nfPArr = len(fParArray)
    fwh = 3
    dim = 2
    npts = 2 ** 6
    nRep = 20
    nPlot = 2
    thetaAll = np.zeros((nrArr, nfPArr))
    rOptAll = np.zeros((nrArr, nfPArr, nRep))
    thOptAll = np.zeros((nrArr, nfPArr, nRep))
    for jjj in range(nrArr):
        for kkk in range(nfPArr):
            thetaAll[jjj, kkk], rOptAll[jjj, kkk, :], thOptAll[jjj, kkk, :], fName = \
                MWE_gaussian_diagnostics_engine(fwh, dim, npts, rArray[jjj], fParArray[:, kkk], nRep, nPlot)

    ## Plot figures for random function
    figH, axH = plt.figure()
    colorArray = ['blue', 'orange', 'green', 'cyan', 'maroon', 'purple']
    nColArray = len(colorArray)
    for jjj in range(nrArr):
        for kkk in range(nfPArr):
            clrInd = np.mod(nfPArr * (jjj) + kkk, nColArray)
            clr = colorArray[clrInd]
            axH.plot(rOptAll[jjj, kkk, :].reshape((nRep, 1)), thOptAll[jjj, kkk, :].reshape((nRep, 1)),
                     marker='.', markersize=20, color=clr)
            # hold on
            axH.scatter(rArray[jjj], thetaAll[jjj, kkk], s=200, c=clr, marker='D')

    axH.set_axis([1, 6, 0.01, 100])
    axH.set_yscale('log')
    axH.set_title(f'd = {dim}, n = {npts}')
    axH.set_xlabel('Inferred $r$)')
    axH.set_ylabel('Inferred $\\theta$')
    figH.savefig(f'{fName}-rthInfer-n-{npts}-d-{dim}.jpg')

    ## Keister example
    fwh = 2
    dim = 3
    npts = 2 ** 6
    nRep = 20
    nPlot = 2
    _, rOptAll, thOptAll, fName = MWE_gaussian_diagnostics_engine(fwh, dim, npts, [], [], nRep, nPlot)

    ## Plot Keister example
    figH = plt.figure()
    plt.plot(rOptAll, thOptAll, marker='.', markersize=20, color='blue')
    # axis([4 6 0.5 1.5])
    # set(gca,'yscale','log')
    plt.xlabel('Inferred $r$')
    plt.ylabel('Inferred $\\theta$')
    plt.title([f'$d = {dim}, n = {npts} $'])
    figH.savefig(f'{fName}-rthInfer-n-{npts}-d-{dim}.jpg')

    ## Keister example
    fwh = 2
    dim = 3
    npts = 2 ** 10
    nRep = 20
    nPlot = 2
    _, rOptAll, thOptAll, fName = MWE_gaussian_diagnostics_engine(fwh, dim, npts,
                                                                  [], [], nRep, nPlot)

    ## Plot Keister example
    figH = plt.figure()
    plt.plot(rOptAll, thOptAll, marker='.', markersize=20, color='blue')
    # axis([4 6 0.5 1.5])
    # set(gca,'yscale','log')
    plt.xlabel('Inferred \(r\)')
    plt.ylabel('Inferred \(\theta\)')
    plt.title(f'$d = {dim}, n = {npts} $')
    figH.savefig(f'{fName}-rthInfer-n-{npts}-d-{dim}.jpg')
