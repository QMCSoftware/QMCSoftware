import unittest
from numpy import arange

from qmcpy.integrand import AsianCall, Keister, Linear
from qmcpy.true_measure import BrownianMotion


class TestAsianCall(unittest.TestCase):

    def test_multi_level_construction(self):
        time_vector = [arange(1 / 4, 5 / 4, 1 / 4),
                       arange(1 / 64, 65 / 64, 1 / 64)]
        dims = [len(tv) for tv in time_vector]
        measure_obj = BrownianMotion(dims, time_vector=time_vector)
        asf = AsianCall(bm_measure=measure_obj)
        self.assertEqual(len(asf), 2)
        self.assertEqual(asf[0].dimension, 4)
        self.assertEqual(asf[1].dimension, 64)


class TestKeister(unittest.TestCase):

    def test_2d_construction(self):
        fun = Keister()
        self.assertEqual(len(fun), 1)
        self.assertEqual(fun[0].dimension, 2)


class TestLinear(unittest.TestCase):

    def test_2d_construction(self):
        fun = Linear()
        self.assertEqual(len(fun), 1)
        self.assertEqual(fun[0].dimension, 2)


if __name__ == "__main__":
    unittest.main()
