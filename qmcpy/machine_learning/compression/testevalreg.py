#!/usr/bin/env python
# coding: utf-8
from approxmeanMXY import *
import matplotlib.pyplot as plt
import numpy as np


# set up random data
N = 1000

x = np.abs(np.random.randn(N, 5))
x = x / (1.01 * np.amax(x, axis=0))
labels = np.random.rand(N, 1)

# test approximation

# dimension of data
d = x.shape[1]

# max number of QMC-points 2^((1+1/alpha)*mmax)
mmax = 5
t = 4

# aux parameters
nsample = 100
errvec = np.zeros((mmax, nsample))


# linear regression function
def f(x, w):
    return np.hstack((np.ones((x.shape[0], 1)), x)) @ w


# loop to plot the approximation error
alpha = 1
for m in range(mmax):
    # compute weights for approximation formula
    weights, z = approxmeanMXY(m+1, int((1 / alpha + 1) * (m+1)) + t, x, labels, alpha)

    for sample in range(nsample):
        # compute exact error for random linear regression functions
        w = np.random.randn(d + 1, 1)

        mval = np.mean((f(x, w) - labels) ** 2)

        # compute approximation
        fz = f(z, w)
        mvalapprox = (weights[:, 0] * (fz ** 2)).sum() - 2 * (weights[:, 1] * fz).sum() + np.mean(labels ** 2)

        # compute relative error
        errvec[m, sample] = abs(mvalapprox - mval) / abs(mval)


# plot errors vs cost
cost = (2 ** ((1 / alpha + 1) * np.array(range(1, mmax+1)))).astype(int)
plt.loglog(cost, np.amax(errvec, axis=1), '-o', cost, np.mean(errvec, axis=1), '-o', cost,  1 / cost ** (1 / (1 + 1 / alpha)), '--k')
plt.show()
