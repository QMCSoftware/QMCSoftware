"""
Single Level Asian Option Pricing
    Run: python workouts/IntegrationExample_AsianOption.py
    Save Output: python workouts/IntegrationExample_AsianOption.py  > outputs/ie_AsianOption.txt
"""

from numpy import arange

from workouts import summary_qmc
from algorithms.stop.CLTStopping import CLTStopping
from algorithms.stop.CLTRep import CLTRep
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.integrate import integrate
from algorithms.integrand.AsianCall import AsianCall
from algorithms.distribution.Measures import *

# IID std_uniform
measureObj = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
OptionObj = AsianCall(measureObj)
distribObj = IIDDistribution(true_distrib=StdUniform(dimension=[64]), seed_rng=7)
stopObj = CLTStopping(distribObj,abs_tol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

# IID std_gaussian
measureObj = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
OptionObj = AsianCall(measureObj)
distribObj = IIDDistribution(true_distrib=StdGaussian(dimension=[64]), seed_rng=7)
stopObj = CLTStopping(distribObj,abs_tol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

# QuasiRandom lattice
measureObj = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
OptionObj = AsianCall(measureObj)
distribObj = QuasiRandom(true_distrib=Lattice(dimension=[64]),seed_rng=7)
stopObj = CLTRep(distribObj,abs_tol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

# QuasiRandom sobol
measureObj = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
OptionObj = AsianCall(measureObj)
distribObj = QuasiRandom(true_distrib=Sobol(dimension=[64]), seed_rng=7)
stopObj = CLTRep(distribObj,abs_tol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

""" Multi-Level Asian Option Pricing """
# IID std_uniform
measureObj = BrownianMotion(time_vector=[arange(1 / 4, 5 / 4, 1 / 4), arange(1 / 16, 17 / 16, 1 / 16), arange(1 / 64, 65 / 64, 1 / 64)])
OptionObj = AsianCall(measureObj)
distribObj = IIDDistribution(true_distrib=StdUniform(dimension=[4, 16, 64]), seed_rng=7)
stopObj = CLTStopping(distribObj,n_max=2**20,abs_tol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

# IID std_gaussian
measureObj = BrownianMotion(time_vector=[arange(1 / 4, 5 / 4, 1 / 4), arange(1 / 16, 17 / 16, 1 / 16), arange(1 / 64, 65 / 64, 1 / 64)])
OptionObj = AsianCall(measureObj)
distribObj = IIDDistribution(true_distrib=StdGaussian(dimension=[4, 16, 64]), seed_rng=7)
stopObj = CLTStopping(distribObj,n_max=2**20,abs_tol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

# QuasiRandom Lattice
measureObj = BrownianMotion(time_vector=[arange(1 / 4, 5 / 4, 1 / 4), arange(1 / 16, 17 / 16, 1 / 16), arange(1 / 64, 65 / 64, 1 / 64)])
OptionObj = AsianCall(measureObj)
distribObj = QuasiRandom(true_distrib=Lattice(dimension=[4,16,64]),seed_rng=7)
stopObj = CLTRep(distribObj,n_max=2**20,abs_tol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

# QuasiRandom sobol
measureObj = BrownianMotion(time_vector=[arange(1 / 4, 5 / 4, 1 / 4), arange(1 / 16, 17 / 16, 1 / 16), arange(1 / 64, 65 / 64, 1 / 64)])
OptionObj = AsianCall(measureObj)
distribObj = QuasiRandom(true_distrib=Sobol(dimension=[4, 16, 64]), seed_rng=7)
stopObj = CLTRep(distribObj,n_max=2**20,abs_tol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)
