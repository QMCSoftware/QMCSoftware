"""
Workout for meanMC_g stopping criterion.
"""
from numpy import arange
from qmcpy import integrate
from qmcpy.discrete_distribution import IIDStdGaussian, IIDStdUniform
from qmcpy.integrand import AsianCall, Keister
from qmcpy.stopping_criterion import MeanMC_g
from qmcpy.true_measure import BrownianMotion, Gaussian

abs_tol = .1

# Keister 3d
integrand = Keister()
discrete_distrib = IIDStdUniform(rng_seed=7)
true_measure = Gaussian(dimension=3, variance=1 / 2)
stopping_criterion = MeanMC_g(discrete_distrib, true_measure, abs_tol=abs_tol)
_, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
data.summarize()

# Singl-Level Asian Option Pricing
time_vec = [arange(1 / 64, 65 / 64, 1 / 64)]
dim = [len(tv) for tv in time_vec]
discrete_distrib = IIDStdGaussian(rng_seed=7)
true_measure = BrownianMotion(dim, time_vector=time_vec)
integrand = AsianCall(true_measure)
stopping_criterion = MeanMC_g(discrete_distrib, true_measure, abs_tol=abs_tol)
_, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
data.summarize()

# Multi-Level Asian Option Pricing
'''
time_vec = [
    arange(1 / 4, 5 / 4, 1 / 4),
    arange(1 / 16, 17 / 16, 1 / 16),
    arange(1 / 64, 65 / 64, 1 / 64)]
dim = [len(tv) for tv in time_vec]
discrete_distrib = IIDStdGaussian(rng_seed=7)
true_measure = BrownianMotion(dim, time_vector=time_vec)
integrand = AsianCall(true_measure)
stopping_criterion = MeanMC_g(discrete_distrib, true_measure, abs_tol=abs_tol)
_, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
data.summarize()
'''
