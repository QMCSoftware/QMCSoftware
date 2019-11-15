""" Unit tests for subclasses of Integrands in QMCPy """

from qmcpy import *
from qmcpy._util import *

import unittest
from numpy import arange


class TestAsianCall(unittest.TestCase):
    """
    Unit tests for AsianCall function in QMCPy.
    """

    def test_multi_level_construction(self):
        time_vector = [arange(1 / 4, 5 / 4, 1 / 4),
                       arange(1 / 64, 65 / 64, 1 / 64)]
        dims = [len(tv) for tv in time_vector]
        measure_obj = BrownianMotion(dims, time_vector=time_vector)
        asf = AsianCall(bm_measure=measure_obj)
        self.assertEqual(len(asf), 2)
        self.assertEqual(asf[0].dimension, 4)
        self.assertEqual(asf[1].dimension, 64)

    def test_mean_type_error(self):
        time_vector = [arange(1 / 4, 5 / 4, 1 / 4)]
        dims = [len(tv) for tv in time_vector]
        measure_obj = BrownianMotion(dims, time_vector=time_vector)
        self.assertRaises(ParameterError, AsianCall, bm_measure=measure_obj, mean_type='misc')


class TestKeister(unittest.TestCase):
    """
    Unit tests for Keister function in QMCPy.
    """

    def test_2d_construction(self):
        fun = Keister()
        self.assertEqual(len(fun), 1)


class TestLinear(unittest.TestCase):
    """
    Unit tests for Linear function in QMCPy.
    """

    def test_2d_construction(self):
        fun = Linear()
        self.assertEqual(len(fun), 1)


if __name__ == "__main__":
    unittest.main()
