import unittest
from numpy import arange
from algorithms.integrand.AsianCall import AsianCall
from algorithms.distribution.Measures import BrownianMotion

class Test_AsianCallFun(unittest.TestCase):

    def test_AsianCallFun_Construction_multi_level(self):
        measureObj = BrownianMotion(timeVector=[arange(1/4,5/4,1/4),arange(1/64,65/64,1/64)])
        asf = AsianCall(BMmeasure=measureObj)
        with self.subTest(): self.assertEqual(len(asf),2)
        with self.subTest(): self.assertEqual(asf[0].dimension,4)
        with self.subTest(): self.assertEqual(asf[1].dimension,64)

if __name__ == "__main__":
    unittest.main()
