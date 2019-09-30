import unittest

from numpy import arange

from algorithms.distribution import Measure
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.function.KeisterFun import KeisterFun
from algorithms.function.AsianCallFun import AsianCallFun
from algorithms.stop.CLTStopping import CLTStopping
from algorithms.stop.CLTRep import CLTRep
from algorithms.integrate import integrate

class IntegrationExampleTest(unittest.TestCase):
    '''
    def test_qmcpy_version(self):
        import qmcpy
        self.assertEqual(qmcpy.__version__, 0.1)
    '''

    def test_KeisterFun_2D(self):
        absTol = .3
        dim = 2
        fun = KeisterFun()
        measureObj = Measure().iid_zmean_gaussian(dimension=[dim],variance=[1 / 2])
        distribObj = IIDDistribution(trueD=Measure().std_gaussian(dimension=[dim]))
        stopObj = CLTStopping(distribObj,nInit=16, absTol=absTol, alpha=.01, inflate=1.2)
        sol, out = integrate(fun, measureObj, distribObj, stopObj)
        true_value = 1.808186429263620
        # In Mathematica (or WolframAlpha):
        # N[Integrate[E^(-x1^2 - x2^2) Cos[Sqrt[x1^2 + x2^2]], {x1, -Infinity, Infinity}, {x2, -Infinity, Infinity}]]
        self.assertTrue(abs(sol-true_value)<absTol)
    
    def test_AsianOption_MultiLevel(self):
        absTol = 0.1
        timeSeries_levels = [arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)]
        measureObj = Measure().brownian_motion(timeVector=timeSeries_levels)
        OptionObj = AsianCallFun(measureObj)
        distribObj = QuasiRandom(trueD=Measure().lattice(dimension=[4,16,64]))
        stopObj = CLTRep(distribObj,absTol=absTol)
        sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
        true_value = 6.20
        self.assertTrue(abs(sol-true_value)<absTol)


if __name__ == "__main__":
    unittest.main()
