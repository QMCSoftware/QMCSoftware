#!/usr/bin/python_prototype/
"""
3 Dimensonal Keister Function
    Run: python workouts/wo_keister.py
    Save Output: python workouts/wo_keister.py  > outputs/ie_KeisterFun.txt
"""

from qmcpy import integrate
from qmcpy._util import summarize
from qmcpy.integrand import Keister
from qmcpy.discrete_distribution import IIDStdGaussian,IIDStdUniform,Lattice,Sobol
from qmcpy.true_distribution import Gaussian
from qmcpy.stop import CLT, CLTRep

def test_distributions_keister():
    dim = 3

    # IID Standard Uniform
    integrand = Keister()
    discrete_distrib = IIDStdUniform()
    true_distrib = Gaussian(dimension=dim,variance=1/2)
    stop = CLT(discrete_distrib,true_distrib, abs_tol=.01)
    sol, data = integrate(integrand, discrete_distrib, true_distrib, stop)
    data.summarize()

    # IID Standard Gaussian
    integrand = Keister()
    discrete_distrib = IIDStdGaussian()
    true_distrib = Gaussian(dimension=dim,variance=1/2)
    stop = CLT(discrete_distrib,true_distrib, abs_tol=.01)
    sol, data = integrate(integrand, discrete_distrib, true_distrib, stop)
    data.summarize()

    # Lattice
    integrand = Keister()
    discrete_distrib = Lattice()
    true_distrib = Gaussian(dimension=dim,variance=1/2)
    stop = CLTRep(discrete_distrib,true_distrib, abs_tol=.01)
    sol, data = integrate(integrand, discrete_distrib, true_distrib, stop)
    data.summarize()

    # Sobol
    integrand = Keister()
    discrete_distrib = Sobol()
    true_distrib = Gaussian(dimension=dim,variance=1/2)
    stop = CLTRep(discrete_distrib,true_distrib, abs_tol=.01)
    sol, data = integrate(integrand, discrete_distrib, true_distrib, stop)
    data.summarize()


test_distributions_keister()