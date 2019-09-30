import numpy as np
import unittest

from algorithms.function.asian_call_integrand import AsianCallFun
from algorithms.distribution import Measure


class TestAsianCallFun(unittest.TestCase):

    def test_AsianCallFun_Construction_multi_level(self):
        # NOTE(Mike) - I feel like this should be linspace, not arange
        time_vector = [np.arange(1 / 4, 5 / 4, 1 / 4), np.arange(1 / 64, 65 / 64, 1 / 64)]
        bm_measure = Measure().brownian_motion(timeVector=time_vector)
        asf = AsianCallFun(bm_measure=bm_measure)
        with self.subTest():
            self.assertEqual(len(asf), 2)
        with self.subTest():
            self.assertEqual(asf[0].dimension, 4)
        with self.subTest():
            self.assertEqual(asf[1].dimension, 64)


if __name__ == "__main__":
    unittest.main()
