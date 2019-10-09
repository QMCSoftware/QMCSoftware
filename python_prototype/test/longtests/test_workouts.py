import unittest

from numpy import abs, arange
from workouts.wo_3d_point_distribution import plot3d
from workouts.wo_abstol_runtime import comp_Clt_vs_cltRep_runtimes
from workouts.wo_asian_option import test_distributions_asian_option
from workouts.wo_keister import test_distributions_keister
from workouts.wo_integrate_default import integrate_default
from workouts.wo_lazy_customizations import custom_lazy_generator,custom_lazy_integrand

class Test_Workouts(unittest.TestCase):
    def test_3d_point_distribution(self):
        plot3d()

    def test_abstol_runtime(self):
        comp_Clt_vs_cltRep_runtimes(arange(0.1, 0.6, 0.1))

    def test_asian_option(self):
        test_distributions_asian_option()

    def test_keister(self):
        test_distributions_keister()

    def test_integrate_default(self):
        sol, data = integrate_default()
        tol = data.stopping_criterion.abs_tol
        self.assertTrue(abs(sol - 1.5) < tol)
    
    def test_custom_customs(self):
        custom_lazy_generator()
        custom_lazy_integrand()


