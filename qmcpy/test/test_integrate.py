import numpy as np
import unittest

from algorithms.distribution import Measure
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.function.keister_integrand import KeisterFun
from algorithms.function.asian_call_integrand import AsianCallFun
from algorithms.stop.CLTStopping import CLTStopping
from algorithms.stop.CLTRep import CLTRep
from algorithms.integrate import integrate


class IntegrationExampleTest(unittest.TestCase):
    '''
    def test_qmcpy_version(self):
        import qmcpy
        self.assertEqual(qmcpy.__version__, 0.1)
    '''

    def test_keister_integrand(self):
        abs_tol = .3
        dim = 2
        integrand = KeisterFun()
        measure = Measure().iid_zmean_gaussian(dimension=[dim], variance=[.5])
        distribution = IIDDistribution(trueD=Measure().std_gaussian(dimension=[dim]))
        stopping_criteria = CLTStopping(distribution, nInit=16, absTol=abs_tol, alpha=.01, inflate=1.2)
        sol, _ = integrate(integrand, measure, distribution, stopping_criteria)
        true_value = 1.80819
        self.assertTrue(abs(sol - true_value) < abs_tol)

    def test_asian_option_multilevel_integrand(self):
        abs_tol = 0.1
        time_series_levels = [
            np.arange(1 / 4, 5 / 4, 1 / 4),
            np.arange(1 / 16, 17 / 16, 1 / 16),
            np.arange(1 / 64, 65 / 64, 1 / 64),
        ]
        measure = Measure().brownian_motion(timeVector=time_series_levels)
        integrand = AsianCallFun(measure)
        distribution = QuasiRandom(trueD=Measure().lattice(dimension=[4, 16, 64]))
        stopping_criteria = CLTRep(distribution, absTol=abs_tol)
        sol, _ = integrate(integrand, measure, distribution, stopping_criteria)
        true_value = 6.20
        self.assertTrue(abs(sol - true_value) < abs_tol)


if __name__ == "__main__":
    unittest.main()
