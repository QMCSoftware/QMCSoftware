""" Unit tests for integrate method in QMCPy """

from qmcpy import *
from numpy import array, inf, pi, sqrt, exp
import sys
vinvo = sys.version_info
if vinvo[0]==3: import unittest
else: import unittest2 as unittest

class IntegrationExampleTest(unittest.TestCase):

    def test_keister(self):
        """
        Mathematica:
        3D:  N[Integrate[E^(-x1^2 - x2^2 - x3^2) Cos[Sqrt[x1^2 + x2^2 + x3^2]], {x1, -Infinity, Infinity}, {x2, -Infinity, Infinity}, {x3, -Infinity, Infinity}]]
        2D:  N[Integrate[E^(-x1^2 - x2^2) Cos[Sqrt[x1^2 + x2^2]], {x1, -Infinity, Infinity}, {x2, -Infinity, Infinity}]]
        1D:  N[Integrate[E^(-x^2) Cos[Sqrt[x^2]], {x, -Infinity, Infinity}]]
        """
        abs_tol = .01
        dimensions = [1, 2, 3]
        true_values = [1.3803884470431430, 1.808186429263620, 2.168309102165481]
        for i in range(len(dimensions)):
            distribution = IIDStdGaussian(dimension=dimensions[i])
            measure = Gaussian(distribution, covariance=1./2)
            integrand = Keister(measure)
            solution,data = CubMCCLT(integrand,abs_tol=abs_tol).integrate()
            self.assertTrue(abs(solution - true_values[i]) < abs_tol)

    def test_asian_option_multi_level(self):
        abs_tol =.05
        distribution = IIDStdGaussian()
        measure = BrownianMotion(distribution)
        integrand = AsianOption(measure,multi_level_dimensions=[4,16,64])
        solution,data = CubMCCLT(integrand, abs_tol).integrate()
        true_value = 1.7845
        self.assertTrue(abs(solution - true_value) < abs_tol)

    def test_lebesgue_bounded_measure(self):
        """ Mathematica: Integrate[x^3 y^3, {x, 1, 3}, {y, 3, 6}] """
        abs_tol = 1
        dimension = 2
        distribution = Sobol(dimension=2, randomize=True, seed=7)
        measure = Lebesgue(distribution, lower_bound=[1,3], upper_bound=[3,6])
        integrand = CustomFun(measure, lambda x: (x.prod(1))**3)
        solution,data = CubQMCSobolG(integrand, abs_tol=abs_tol).integrate()
        true_value = 6075
        self.assertTrue(abs(solution - true_value) < abs_tol)
    
    def test_lebesgue_inf_measure(self):
        abs_tol = .1
        distribution = Lattice(1)
        measure = Lebesgue(distribution, lower_bound=-inf, upper_bound=inf)
        integrand = CustomFun(measure, lambda x: exp(-x**2))
        solution,data = CubQMCLatticeG(integrand,abs_tol=abs_tol).integrate()
        true_value = sqrt(pi)
        self.assertTrue(abs(solution - solution) < abs_tol)
    
    def test_lebesgue_inf_measure_2d(self):
        abs_tol = .1
        distribution = Lattice(2)
        measure = Lebesgue(distribution, lower_bound=-inf, upper_bound=inf)
        integrand = CustomFun(measure, lambda x: exp(-x**2).prod(1))
        solution,data = CubQMCCLT(integrand,abs_tol=abs_tol).integrate()
        true_value = pi
        self.assertTrue(abs(solution - solution) < abs_tol)

    def test_uniform_measure(self):
        """ Mathematica: Integrate[(x^3 y^3)/6, {x, 1, 3}, {y, 3, 6}] """
        abs_tol = 1
        dimension = 2
        distribution = Lattice(dimension=2, randomize=True, order='mps')
        measure = Uniform(distribution, lower_bound=[1,3], upper_bound=[3,6])
        integrand = CustomFun(measure, lambda x: (x.prod(1))**3)
        solution,data = CubQMCCLT(integrand, abs_tol=abs_tol).integrate()
        true_value = 6075 / 6
        self.assertTrue(abs(solution - true_value) < abs_tol)

    def test_linear(self):
        """ Mathematica:
        1D: Integrate[x, {x, 0, 1}]
        2D: Integrate[x+y, {x, 0, 1}, {y, 0, 1}]
        3D: Integrate[x+y+z, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]
        """
        abs_tol = .01
        dimensions = [1, 2, 3]
        true_values = [0.5, 1, 1.5]
        for i in range(len(dimensions)):
            distribution = Sobol(dimension=dimensions[i], randomize=True)
            measure = Uniform(distribution)
            integrand = Linear(measure)
            solution,data = CubQMCCLT(integrand, abs_tol=abs_tol).integrate()
            self.assertTrue(abs(solution - true_values[i]) < abs_tol)

    def test_quick_construct(self):
        """
        Infer true measure's dimension from integrand's

        Mathematica:
        1D: Integrate[5x, {x, 0, 1}]
        2D: Integrate[5(x+y), {x, 0, 1}, {y, 0, 1}]
        3D: Integrate[5(x+y+z), {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]
        """
        abs_tol = .01
        dimensions = [1, 2, 3]
        true_values = [2.5, 5, 7.5]
        for i in range(len(dimensions)):
            distribution = IIDStdUniform(dimension=dimensions[i])
            measure = Uniform(distribution)
            integrand = CustomFun(measure, lambda x: (5*x).sum(1))
            solution,data = CubMCG(integrand, abs_tol=abs_tol).integrate()
            self.assertTrue(abs(solution - true_values[i]) < abs_tol)

    def test_quick_construct2(self):
        """
        Test integrands with parameters

        Mathematica: integrate[b*(x-a)^2, {x,1,0}]
        """
        abs_tol = .01
        a_list = [1., 2.]
        b_list = [4., 5.]
        true_values = [(b / 3) * (3 * a * (a - 1) + 1) for a, b in zip(a_list, b_list)]
        for i in range(2):
            a_i = a_list[i]
            b_i = b_list[i]
            distribution = Lattice(dimension=1, randomize=True)
            measure = Uniform(distribution)
            integrand = CustomFun(measure, lambda x, a=a_i, b=b_i: b * (x - a) ** 2)
            solution,data = CubQMCLatticeG(integrand, abs_tol=abs_tol).integrate()
            self.assertTrue(abs(solution - true_values[i]) < abs_tol)

    def test_european_call(self):
        abs_tol = 1e-2
        ddistrib = Sobol(dimension=16, seed=7)
        measure = BrownianMotion(ddistrib)
        integrand = EuropeanOption(measure,
            volatility = .2,
            start_price = 5,
            strike_price = 10,
            interest_rate = .01,
            call_put = 'call')
        algorithm = CubQMCSobolG(integrand, abs_tol)
        solution,data = algorithm.integrate()
        true_value = integrand.get_exact_value()
        self.assertTrue(abs(solution-true_value) < abs_tol)
    
    def test_european_put(self):
        abs_tol = 1e-2
        ddistrib = Lattice(dimension=16, seed=17)
        measure = BrownianMotion(ddistrib)
        integrand = EuropeanOption(measure,
            volatility = .5,
            start_price = 10,
            strike_price = 10,
            interest_rate = .01,
            call_put = 'put')
        algorithm = CubQMCLatticeG(integrand, abs_tol)
        solution,data = algorithm.integrate()
        true_value = integrand.get_exact_value()
        self.assertTrue(abs(solution-true_value) < abs_tol)

    def test_european_put_bayes(self):
        abs_tol = 1e-2
        ddistrib = Lattice(dimension=16, seed=17, order='linear', randomize=False)
        measure = BrownianMotion(ddistrib)
        integrand = EuropeanOption(measure,
            volatility = .5,
            start_price = 10,
            strike_price = 10,
            interest_rate = .01,
            call_put = 'put')
        algorithm = CubBayesLatticeG(integrand, abs_tol, ptransform='Baker')
        solution,data = algorithm.integrate()
        true_value = integrand.get_exact_value()
        self.assertTrue(abs(solution-true_value) < abs_tol)

if __name__ == "__main__":
    unittest.main()
