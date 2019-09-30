import unittest

from algorithms.distribution.Measures import StdGaussian
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.distribution import MeasureCompatibilityError


class Test_IIDDistribution(unittest.TestCase):

    def test_IIDGen_in_QuasiClass(self):
        self.assertRaises(MeasureCompatibilityError,QuasiRandom,trueD=StdGaussian([2]))

if __name__ == "__main__":
    unittest.main()