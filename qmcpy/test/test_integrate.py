import unittest

from numpy import arange

from algorithms.distribution.Measures import IIDZeroMeanGaussian,StdGaussian,BrownianMotion,Lattice
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.integrand.Keister import Keister
from algorithms.integrand.AsianCall import AsianCall
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
        abs_tol = .3
        dim = 2
        fun = Keister()
        measureObj = IIDZeroMeanGaussian(dimension=[dim],variance=[1 / 2])
        distribObj = IIDDistribution(true_distrib=StdGaussian(dimension=[dim]))
        stopObj = CLTStopping(distribObj,n_init=16, abs_tol=abs_tol, alpha=.01, inflate=1.2)
        sol, out = integrate(fun, measureObj, distribObj, stopObj)
        true_value = 1.808186429263620
        # In Mathematica (or WolframAlpha):
        # N[Integrate[E^(-x1^2 - x2^2) Cos[Sqrt[x1^2 + x2^2]], {x1, -Infinity, Infinity}, {x2, -Infinity, Infinity}]]
        self.assertTrue(abs(sol-true_value)<abs_tol)
    
    def test_AsianOption_MultiLevel(self):
        abs_tol = 0.1
        timeSeries_levels = [arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)]
        measureObj = BrownianMotion(time_vector=timeSeries_levels)
        OptionObj = AsianCall(measureObj)
        distribObj = QuasiRandom(true_distrib=Lattice(dimension=[4,16,64]))
        stopObj = CLTRep(distribObj,abs_tol=abs_tol)
        sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
        true_value = 6.20
        self.assertTrue(abs(sol-true_value)<abs_tol)


if __name__ == "__main__":
    unittest.main()
