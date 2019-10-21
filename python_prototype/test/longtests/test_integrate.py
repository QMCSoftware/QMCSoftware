import unittest
from numpy import array, arange

from qmcpy import integrate
from qmcpy.discrete_distribution import IIDStdGaussian, Lattice
from qmcpy.true_measure import Gaussian, BrownianMotion, Lebesgue
from qmcpy.integrand import Keister, AsianCall, QuickConstruct
from qmcpy.stopping_criterion import CLT, CLTRep


class IntegrationExampleTest(unittest.TestCase):

    def test_keister_2d(self):
        """ Mathematica: N[Integrate[E^(-x1^2 - x2^2) Cos[Sqrt[x1^2 + x2^2]], {x1, -Infinity, Infinity}, {x2, -Infinity, Infinity}]] """
        abs_tol = .1
        integrand = Keister()
        discrete_distrib = IIDStdGaussian()
        true_measure = Gaussian(dimension=2, variance=1 / 2)
        stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=abs_tol)
        sol, _ = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
        true_value = 1.808186429263620

        self.assertTrue(abs(sol - true_value) < abs_tol)

    def test_asian_option_multi_level(self):
        abs_tol = 0.1
        time_vec = [
            arange(1 / 4, 5 / 4, 1 / 4),
            arange(1 / 16, 17 / 16, 1 / 16),
            arange(1 / 64, 65 / 64, 1 / 64)]
        dims = [len(tv) for tv in time_vec]
        discrete_distrib = Lattice()
        true_measure = BrownianMotion(dims, time_vector=time_vec)
        integrand = AsianCall(true_measure)
        stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol)
        sol, _ = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
        true_value = 6.20
        self.assertTrue(abs(sol - true_value) < abs_tol)

    def test_Lebesgue_measure(self):
        """ Mathematica: Integrate[5 x^3 y^3, {x, 1, 3}, {y, 3, 6}] """
        abs_tol = 1
        integrand = QuickConstruct(custom_fun=lambda x: 5 * (x.prod(1))**3)
        true_measure = Lebesgue(
            dimension=[2],
            uniform_lower_bound=[array([1, 3])],
            uniform_upper_bound=[array([3, 6])])
        discrete_distrib = Lattice(rng_seed=7)
        stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol)
        sol, _ = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)

        print(sol)
        true_value = 30375
        self.assertTrue(abs(sol - true_value) < abs_tol)


if __name__ == "__main__":
    unittest.main()
