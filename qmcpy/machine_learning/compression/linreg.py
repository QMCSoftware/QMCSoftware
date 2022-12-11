#!/usr/bin/env python
# coding: utf-8

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from ctypes import *
from numpy.ctypeslib import ndpointer
from sklearn.preprocessing import MinMaxScaler
import logging



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
	lib = cdll.LoadLibrary("../c_lib/c_lib.cpython-39-darwin.so")
	computeWeights = lib.computeWeights
	computeWeights.restype = ndpointer(dtype=c_double, shape=(1 + outs, Nqmc))

	# compute weights
	logging.info(f"\n{nu = }, {m = }, {s = }, {Ndata = }, {Nqmc = }, {outs = }")
	weights = computeWeights(c_int(nu), c_int(m), c_int(s), c_int(Ndata), c_int(Nqmc), c_int(outs),
                         c_void_p(X.ctypes.data), c_void_p(qmc_points.ctypes.data), c_void_p(y.ctypes.data))
	weights = np.transpose(weights)
	print(f"{weights.shape = }")
	weights=computeMXYmex(nu,m,base,x,z,y)
	regr = LinearRegression()
	mod = regr.fit(weights[:, :-1], weights[:, -1])
	yfit = mod.predict(weights[:, :-1])
	mse = mean_squared_error(weights[:, -1], yfit)
	logging.info(f"Compressed data {mse = }")
	logging.info(f"{mod.coef_ = }")
	
	# load QMC points
	m, nu = 10, 3
	Nqmc = 100
	outs = s
	qmc_points = np.loadtxt('sobol.dat')  # 4097 x 100
	qmc_points = qmc_points[0:Nqmc, 0:s]  # Nqmc x s



	"""
	mse = 0.08328613078165403
	mod.coef_ = array([-0.001,  0.004, -0.002,  0.002,  0.001, -0.001])
	
	Weighted mse = 0.08329050285745367
	sample_weight.shape = (1000000,)
	mod.coef_ = array([-0.002,  0.004, -0.002,  0.002,  0.001, -0.001])
	
	nu = 3, m = 10, s = 6, Ndata = 1000000, Nqmc = 100, outs = 6
	weights.shape = (100, 7)
	Compressed data mse = 1.6540768# load QMC points
	m, nu = 10, 3
	Nqmc = 100
	outs = s
	qmc_points = np.loadtxt('sobol.dat')  # 4097 x 100
	qmc_points = qmc_points[0:Nqmc, 0:s]  # Nqmc x s

	#function [weights,z] = approxmeanMXY(nu,m,x,y,d)
	#s=size(x,2);
	#base=2;
	
	def approxmeanMXY(nu,m,x,y,d):
		size()
	def MyHOSobol(m,d,s=2):
		pass
		#base=2;
	'''z=MyHOSobol(m,s,d);
	#weights=computeMXYmex(nu,m,base,x,z',y);
	
	#weights = mexFunction(c_int(m), c_int(mp), c_int(base), c_double(), c_int(outs),
	#                         c_void_p(X.ctypes.data), c_void_p(qmc_points.ctypes.data), c_void_p(y.ctypes.data))
	
	#weights=computeMXYmex(nu,m,base,x,z',y);
	
	#function [weights,z] = approxmeanMXY(nu,m,x,y,d)
	#s=size(x,2);#function [weights,z] = approxmeanMXY(nu,m,x,y,d)
	#s=size(x,2);
	#base=2;
	#z=MyHOSobol(m,s,d);function [weights,z] = approxmeanMXY(nu,m,x,y,d)
		  
	#function [weights,z] = approxmeanMXY(nu,m,x,y,d)
	#s=size(x,2);
	#base=2;
	#z=MyHOSobol(m,s,d);
	#weights=computeMXYmex(nu,m,base,x,z',y);

	#weights = mexFunction(c_int(m), c_int(mp), c_int(base), c_double(), c_int(outs),
	#                         c_void_p(X.ctypes.data), c_void_p(qmc_points.ctypes.data), c_void_p(y.ctypes.data))
	
	#weights=computeMXYmex(nu,m,base,x,z',y);

	#weights = mexFunction(c_int(m), c_int(mp), c_int(base), c_double(), c_int(outs),
	#                         c_void_p(X.ctypes.data), c_void_p(qmc_points.ctypes.data), c_void_p(y.ctypes.data))



	# # The weighted model
	# regr = LinearRegression()
	# mod = regr.fit(X_minmax, y, sample_weight)
	# yfit = mod.predict(X_minmax)
	# mse = mean_squared_error(y, yfit, sample_weight=sample_weight)
	# logging.info(f"\nWeighted {mse = }")
	# logging.info(f"{sample_weight.shape = }")
	# logging.info(f"{mod.coef_ = }")

	# load c functions
	lib = cdll.LoadLibrary("../c_lib/c_lib.cpython-39-darwin.so")
	computeWeights = lib.computeWeights
	computeWeights.restype = ndpointer(dtype=c_double, shape=(1 + outs, Nqmc))

	# compute weights
	logging.info(f"\n{nu = }, {m = }, {s = }, {Ndata = }, {Nqmc = }, {outs = }")

	def approxmeanMXY(nu,m,x,y,d):
	    size()
	def MyHOSobol(m,s=2,d):
		pass
		  
	weights = np.transpose(weights)
	print(f"{weights.shape = }")
	
	# The compressed data model
	regr = LinearRegression()
	mod = regr.fit(weights[:, :-1], weights[:, -1])
	yfit = mod.predict(weights[:, :-1])
	mse = mean_squared_error(weights[:, -1], yfit)
	logging.info(f"Compressed data {mse = }")
	logging.info(f"{mod.coef_ = }")
	
	mse = 0.08328613078165403
	mod.coef_ = array([-0.001,  0.004, -0.002,  0.002,  0.001, -0.001])
	
	Weighted mse = 0.08329050285745367
	sample_weight.shape = (1000000,)
	mod.coef_ = array([-0.002,  0.004, -0.002,  0.002,  0.001, -0.001])
	
	nu = 3, m = 10, s = 6, Ndata = 1000000, Nqmc = 100, outs = 6
	weights.shape = (100, 7)
	Compressed data mse = 1.6540768918948703e-14
	mod.coef_ = array([-0.022,  0.677, -0.653,  0.492,  0.207,  0.339])
	
	#function [weights,z] = approxmeanMXY(nu,m,x,y,d)
	#s=size(x,2);
	#base=2;
	#z=MyHOSobol(m,s,d);function [weights,z] = approxmeanMXY(nu,m,x,y,d)
	"""
if __name__ == '__main__':
    main()
