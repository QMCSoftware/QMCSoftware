import unittest
from numpy import arange

from qmcpy import integrate
from qmcpy.discrete_distribution import IIDStdGaussian, Lattice
from qmcpy.true_measure import Gaussian, BrownianMotion
from qmcpy.integrand import Keister, AsianCall
from qmcpy.stop import CLT, CLTRep

class IntegrationExampleTest(unittest.TestCase):

    def test_Keister_2D(self):
        # IID Standard Uniform
        abs_tol = .1
        integrand = Keister()
        discrete_distrib = IIDStdGaussian()
        true_measure = Gaussian(dimension=2, variance=1/2)
        stop = CLT(discrete_distrib, true_measure, abs_tol=abs_tol)
        sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
        true_value = 1.808186429263620
        # In Mathematica (or WolframAlpha):
        # N[Integrate[E^(-x1^2 - x2^2) Cos[Sqrt[x1^2 + x2^2]], {x1, -Infinity, Infinity}, {x2, -Infinity, Infinity}]]
        self.assertTrue(abs(sol - true_value) < abs_tol)

    def test_AsianOption_MultiLevel(self):
        abs_tol = 0.1
        time_vec = [
            arange(1 / 4, 5 / 4, 1 / 4),
            arange(1 / 16, 17 / 16, 1 / 16),
            arange(1 / 64, 65 / 64, 1 / 64)]
        dims = [len(tv) for tv in time_vec]
        discrete_distrib = Lattice()
        true_measure = BrownianMotion(dims, time_vector=time_vec)
        integrand = AsianCall(true_measure)
        stop = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol)
        sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
        true_value = 6.20
        self.assertTrue(abs(sol - true_value) < abs_tol)


if __name__ == "__main__":
    unittest.main()
