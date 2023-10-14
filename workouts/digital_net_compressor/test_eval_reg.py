"""
The purpose of this script is to conditionally reproduce Figure 3 in the paper:

Dick, Josef, and Michael Feischl. "A quasi-Monte Carlo data compression algorithm
for machine learning." Journal of Complexity 67 (2021): 101587.

"""
import os
import matplotlib.pyplot as plt
import numpy as np
from qmcpy import DigitalNetDataCompressor


# linear regression function
def f(x, w):
    return np.hstack((np.ones((x.shape[0], 1)), x)) @ w


def data_compress_lin_reg(x, labels, mmax=5, nsample=100, alpha=1):
    # dimension of data
    d = x.shape[1]
    t = 4
    # aux parameters
    err_vec = np.zeros((mmax, nsample))
    path = os.path.dirname(__file__) + os.sep
    # loop to plot the approximation error
    for m in range(mmax):
        # compute weights for approximation formula
        dn = DigitalNetDataCompressor(nu=m + 1, m=int(np.ceil((1 + 1.0 / alpha) * (m + 1) + t)), dataset=x,
                                      labels=labels,
                                      alpha=alpha, sobol=np.loadtxt(f"{path}sobol2.dat"))
        dn.approx_mean_mxy()
        weights, z = dn.weights, dn.sobol

        # compute exact error for random linear regression functions
        w = np.random.randn(d + 1, nsample)
        mval = np.mean((f(x, w) - labels.reshape(-1, 1)) ** 2, axis=0)

        # compute approximation
        fz = f(z, w)
        mvalapprox = (weights[:, 0].reshape(-1, 1) * (fz ** 2)).sum(axis=0) - 2 * (
                weights[:, 1].reshape(-1, 1) * fz).sum(axis=0) + np.mean(labels ** 2)

        # compute relative error
        err_vec[m, :] = np.abs(mvalapprox - mval) / np.abs(mval)

    # compute errors vs cost_vec
    cost_vec = 2 ** np.ceil((1.0 / alpha + 1) * np.array(range(1, mmax + 1)))

    return cost_vec, err_vec


if __name__ == "__main__":
    # set up random data
    N = int(1e4)  # 1e6
    # np.random.seed(2)
    x = np.abs(np.random.randn(N, 5))
    x = x / (1.01 * np.amax(x, axis=0))
    labels = np.random.rand(N, 1).reshape(-1)
    alpha_vec = [1]  # [1, 2]
    mmax_vec = [5]  # [5, 7]
    plt.figure(figsize=(10, 8))
    for alpha, mmax in zip(alpha_vec, mmax_vec):
        cost, errvec = data_compress_lin_reg(x, labels, mmax=mmax, alpha=alpha)
        if alpha == 1:
            marker1, marker2, marker3 = '-ob', '-xr', '--g'
        elif alpha == 2:
            marker1, marker2, marker3 = '-dy', '-sm', '--k'
        plt.loglog(cost, np.amax(errvec, axis=1), marker1, markerfacecolor='none',
                   label=r"$\alpha$ = " + str(alpha) + ", maximum")
        plt.loglog(cost, np.mean(errvec, axis=1), marker2, markerfacecolor='none',
                   label=r"$\alpha$ = " + str(alpha) + ", average")
        plt.loglog(cost, 1 / cost ** (1.0 / (1 + 1 / alpha)), marker3,
                   label=r"$\alpha$ = " + str(alpha) + ", predicted convergence order")

    plt.xlabel("Cost")
    plt.ylabel("Error")
    plt.ylim(1e-3, 10)  # set labels-axis limits
    plt.legend(loc='lower left')  # bbox_to_anchor=(1, 0.5))  # display the legends
    plt.title("QMC Data Compression using QMCPy for Linear Regression")
    plt.tight_layout()
    plt.savefig("test_eval_reg.png")
    plt.show(block=False)
    #plt.pause(20)  # show plot for n seconds