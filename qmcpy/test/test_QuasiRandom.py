import unittest

from algorithms.distribution import Measure
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.distribution import MeasureCompatibilityError


class Test_IIDDistribution(unittest.TestCase):

    def test_IIDGen_in_QuasiClass(self):
        self.assertRaises(MeasureCompatibilityError,QuasiRandom,trueD=Measure().std_gaussian())

if __name__ == "__main__":
    unittest.main()