import unittest

from numpy import arange

from algorithms.distribution import measure
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.function.KeisterFun import KeisterFun
from algorithms.function.AsianCallFun import AsianCallFun
from algorithms.stop.CLTStopping import CLTStopping
from algorithms.stop.CLT_Rep import CLT_Rep
from algorithms.integrate import integrate

class IntegrationExampleTest(unittest.TestCase):
    '''
    def test_qmcpy_version(self):
        import qmcpy
        self.assertEqual(qmcpy.__version__, 0.1)
    '''

    def test_KeisterFun(self):
        absTol = .3
        dim = 2
        fun = KeisterFun()
        measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[1 / 2])
        distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[dim]))
        stopObj = CLTStopping(distribObj,nInit=16, absTol=absTol, alpha=.01, inflate=1.2)
        sol, out = integrate(fun, measureObj, distribObj, stopObj)
        true_value = 1.80819
        self.assertTrue(abs(sol-true_value)<absTol)
    
    def test_AsianOption_MultiLevel(self):
        absTol = 0.1
        timeSeries_levels = [arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)]
        measureObj = measure().BrownianMotion(timeVector=timeSeries_levels)
        OptionObj = AsianCallFun(measureObj)
        distribObj = QuasiRandom(trueD=measure().lattice(dimension=[4,16,64]))
        stopObj = CLT_Rep(distribObj,absTol=absTol)
        sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
        true_value = 6.20
        self.assertTrue(abs(sol-true_value)<absTol)


if __name__ == "__main__":
    unittest.main()
