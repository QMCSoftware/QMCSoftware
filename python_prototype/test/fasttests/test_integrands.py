""" Unit tests for subclasses of Integrands in QMCPy """

import unittest

from numpy import arange
from qmcpy import *
from qmcpy.util import *


class TestAsianCall(unittest.TestCase):
    """
    Unit tests for AsianCall function in QMCPy.
    """

    def test_f(self):
        distribution = Sobol(dimension=4)
        measure = BrownianMotion(distribution, time_vector=[1/4,1/2,3/4,1])
        integrand = AsianCall(measure)
        integrand.f(distribution.gen_samples(n_min=0,n_max=4))

class TestKeister(unittest.TestCase):
    """
    Unit tests for Keister function in QMCPy.
    """

    def test_f(self):
        distribution = Sobol(dimension=3)
        measure = Uniform(distribution)
        integrand = Keister(measure)
        integrand.f(distribution.gen_samples(n_min=0,n_max=4))


class TestLinear(unittest.TestCase):
    """
    Unit tests for Linear function in QMCPy.
    """

    def test_f(self):
        distribution = Sobol(dimension=3)
        measure = Uniform(distribution)
        integrand = Linear(measure)
        integrand.f(distribution.gen_samples(n_min=0,n_max=4))
    
class TestQuickConstruct(unittest.TestCase):
    """
    Unit tests for QuickConstruct function in QMCPy.
    """

    def test_f(self):
        distribution = Sobol(dimension=3)
        measure = Uniform(distribution)
        integrand = QuickConstruct(measure, lambda x: x.sum(1))
        integrand.f(distribution.gen_samples(n_min=0,n_max=4))


if __name__ == "__main__":
    unittest.main()
