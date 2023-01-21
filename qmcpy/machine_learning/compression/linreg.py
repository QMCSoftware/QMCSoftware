#!/usr/bin/env python
# coding: utf-8

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from ctypes import *
from numpy.ctypeslib import ndpointer
from sklearn.preprocessing import MinMaxScaler
import logging
from myhosobol import MyHOSobol
import math
def approxmeanMXY(nu, m, x, y, d):
    base = 2
    s = x.shape[1]
    MyHOSobol(m,s,d)
    weights = MyHOSobol(m,s,d)

def firstMissingPositive(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    bits = 0
    for num in nums:
        if 0 < num <= len(nums):
            bits |= 1 << (num - 1)

    ret = 1
    while bits & 1 != 0:
        bits >>= 1
        ret += 1
    return ret



def main():
	np.set_printoptions(precision=3)

	# Create a random dataset for testing.
	mu, sigma = 0, 1
	Ndata, s = 1000000, 6
	X = np.random.default_rng().normal(mu, sigma, size=(Ndata, s))
	y = np.random.default_rng().uniform(mu, sigma, Ndata)
	n_samples = X.shape[0]
	
	min_max_scaler = MinMaxScaler()
	X_minmax = min_max_scaler.fit_transform(X)
	
	# Create equal weights except for the last ten
	sample_weight = np.ones(n_samples) * 20
	sample_weight[-10:] *= 30
	sample_weight = sample_weight / sample_weight.max()
	
	# The unweighted model
	print("Fitting the unweighted model")
	regr = LinearRegression()
	mod = regr.fit(X_minmax, y)
	yfit = mod.predict(X_minmax)
	print(f"{yfit = }")
	mse = mean_squared_error(y, yfit)
	print(f"{mse = }")
	# load QMC points
	m, nu = 10, 3
	Nqmc = 100
	outs = s
	qmc_points = np.loadtxt('sobol.dat')  # 4097 x 100
	qmc_points = qmc_points[0:Nqmc, 0:s]  # Nqmc x s


	print(f"{mod.coef_ = }")


	# The compressed data model
	# load c functions
	#lib = cdll.LoadLibrary("../c_lib/c_lib.cpython-39-darwin.so")
	computeWeights = lib.mexFunction
	computeWeights.restype = ndpointer(dtype=c_double, shape=(1 + outs, Nqmc))

	# compute weights
	logging.info(f"\n{nu = }, {m = }, {s = }, {Ndata = }, {Nqmc = }, {outs = }")
	weights = computeWeights(c_int(nu), c_int(m), c_int(s), c_int(Ndata), c_int(Nqmc), c_int(outs), c_void_p(X.ctypes.data), c_void_p(qmc_points.ctypes.data), c_void_p(y.ctypes.data))
	weights = np.transpose(weights)
	print(f"{weights.shape = }")
	weights=computeMXYmex(nu,m,base,x,z,y)
	regr = LinearRegression()
	mod = regr.fit(weights[:, :-1], weights[:, -1])
	yfit = mod.predict(weights[:, :-1])
	mse = mean_squared_error(weights[:, -1], yfit)
	logging.info(f"Compressed data {mse = }")
	logging.info(f"{mod.coef_ = }")

if __name__ == '__main__':
    main()
