import unittest

from algorithms.measures.measures import StdGaussian
from algorithms.distribution.quasi_random import QuasiRandom
from algorithms.distribution import MeasureCompatibilityError

class Test_IIDDistribution(unittest.TestCase):

    def test_IIDGen_in_QuasiClass(self):
        self.assertRaises(MeasureCompatibilityError,QuasiRandom,true_distribution=StdGaussian([2]))

if __name__ == "__main__":
    unittest.main()