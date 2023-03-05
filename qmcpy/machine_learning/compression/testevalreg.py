#!/usr/bin/env python
# coding: utf-8
from approxmeanMXY import *
import matplotlib.pyplot as plt
import numpy as np

def data_compress_lin_reg(x, labels, mmax=5, nsample=100, alpha=1):

    # dimension of data
    d = x.shape[1]

    t = 4

    # aux parameters
    errvec = np.zeros((mmax, nsample))

    # linear regression function
    def f(x, w):
        return np.hstack((np.ones((x.shape[0], 1)), x)) @ w

    # loop to plot the approximation error
    for m in range(mmax):
        # compute weights for approximation formula
        weights, z = approxmeanMXY(m + 1, int(np.ceil((1 + 1.0 / alpha) * (m + 1) + t)), x, labels, alpha)

        # compute exact error for random linear regression functions
        w = np.random.randn(d + 1, nsample)
        mval = np.mean((f(x, w) - labels.reshape(-1, 1)) ** 2, axis=0)

        # compute approximation
        fz = f(z, w)
        mvalapprox = (weights[:, 0].reshape(-1, 1) * (fz ** 2)).sum(axis=0) - 2 * (weights[:, 1].reshape(-1, 1) * fz).sum(axis=0) + np.mean(labels ** 2)

        # compute relative error
        errvec[m,:] = np.abs(mvalapprox - mval) / np.abs(mval)

    # compute errors vs cost
    cost = 2 ** np.ceil((1.0 / alpha + 1) * np.array(range(1, mmax + 1)))

    return cost, errvec

if __name__ == "__main__":

    # set up random data
    N = int(1e6)
    # np.random.seed(2)
    x = np.abs(np.random.randn(N, 5))
    x = x / (1.01 * np.amax(x, axis=0))
    labels = np.random.rand(N, 1).reshape(-1)

    alpha_vec = [1, 2]
    mmax_vec = [5, 7]
    for alpha, mmax in zip(alpha_vec, mmax_vec):
        cost, errvec = data_compress_lin_reg(x, labels, mmax=mmax, alpha=alpha)
        if alpha == 1: marker1, marker2, marker3 = '-ob', '-xr', '--g'
        elif alpha == 2: marker1, marker2, marker3 = '-dy', '-sm', '--k'
        plt.loglog(cost, np.amax(errvec, axis=1), marker1, markerfacecolor='none', label=r"$\alpha$ = "+ str(alpha)+", maximum error")
        plt.loglog(cost, np.mean(errvec, axis=1), marker2, markerfacecolor='none', label=r"$\alpha$ = "+ str(alpha)+", average error")
        plt.loglog(cost, 1 / cost ** (1.0 / (1 + 1 / alpha)), marker3, label=r"$\alpha$ = "+ str(alpha)+", predicted convergence order")

    plt.xlabel("Cost")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # display the legends
    plt.tight_layout()
    plt.savefig("testevalreg.png")
    plt.show(block=False)
    plt.pause(20)  # show plot for n seconds