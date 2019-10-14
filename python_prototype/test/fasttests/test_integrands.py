import unittest
from numpy import arange

from qmcpy.integrand import AsianCall,Keister,Linear
from qmcpy.true_measure import BrownianMotion

class Test_AsianCall(unittest.TestCase):
    
    def test_multi_level_construction(self):
        time_vector = [arange(1/4, 5/4, 1/4), arange(1/64, 65/64, 1/64)]
        dims = [len(tv) for tv in time_vector]
        measureObj = BrownianMotion(dims,time_vector=time_vector)
        asf = AsianCall(bm_measure=measureObj)
        with self.subTest():
            self.assertEqual(len(asf), 2)
        with self.subTest():
            self.assertEqual(asf[0].dimension, 4)
        with self.subTest():
            self.assertEqual(asf[1].dimension, 64)

class Test_Keister(unittest.TestCase):
    
    def test_2d_construction(self):
        fun = Keister()
        with self.subTest():
            self.assertEqual(len(fun), 1)
        with self.subTest():
            self.assertEqual(fun[0].dimension, 2)

class Test_Linear(unittest.TestCase):
    
    def test_2d_construction(self):
        fun = Linear()
        with self.subTest():
            self.assertEqual(len(fun), 1)
        with self.subTest():
            self.assertEqual(fun[0].dimension, 2)

if __name__ == "__main__":
    unittest.main()