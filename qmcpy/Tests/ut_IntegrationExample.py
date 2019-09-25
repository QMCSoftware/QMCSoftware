import unittest

from algorithms.distribution import measure
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.function.KeisterFun import KeisterFun
from algorithms.integrate import integrate
from algorithms.stop.CLTStopping import CLTStopping


class IntegrationExampleTest(unittest.TestCase):

    def test_qmcpy_version(self):
        import qmcpy
        self.assertEqual(qmcpy.__version__, 0.1)


    def test_integral_values(self):
        dim = 2
        fun = KeisterFun()
        measureObj = measure().IIDZMeanGaussian(dimension=[dim],
                                                variance=[1 / 2])
        distribObj = IIDDistribution(
            trueD=measure().stdGaussian(dimension=[dim]),rngSeed=7)
        fun = fun.transformVariable(measureObj, distribObj)
        absTol = .3
        stopObj = CLTStopping(nInit=16, absTol=absTol, alpha=.01, inflate=1.2)
        sol, out = integrate(KeisterFun(), measureObj, distribObj, stopObj)

        true_value = 1.80819
        self.assertEqual(abs(sol - true_value) < absTol, True)


if __name__ == "__main__":
    unittest.main()
