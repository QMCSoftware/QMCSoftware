import unittest

from numpy import random

from algorithms.CLTStopping import CLTStopping
from algorithms.IIDDistribution import IIDDistribution
from algorithms.KeisterFun import KeisterFun
from algorithms.integrate import integrate
from algorithms.measure import measure


class IntegrationExampleTest(unittest.TestCase):

    def testIntegralValues(self):
        random.seed(7)
        dim = 2
        fun = KeisterFun()
        measureObj = measure().IIDZMeanGaussian(dimension=[dim],
                                                variance=[1 / 2])
        distribObj = IIDDistribution(
            trueD=measure().stdGaussian(dimension=[dim]))
        fun = fun.transformVariable(measureObj, distribObj)
        absTol = .3
        stopObj = CLTStopping(nInit=16, absTol=absTol, alpha=.01, inflate=1.2)
        sol, out = integrate(KeisterFun(), measureObj, distribObj, stopObj)

        true_value = 1.80819
        self.assertEqual(abs(sol - true_value) < absTol, True)


if __name__ == "__main__":
    unittest.main()
