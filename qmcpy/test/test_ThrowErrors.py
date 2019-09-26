import unittest

from algorithms.distribution import measure
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution.QuasiRandom import QuasiRandom

from algorithms import DistributionCompatibilityError


class ThrowErrorsTest(unittest.TestCase):

    def test_QuasiGen_in_IIDClass(self):
        self.assertRaises(DistributionCompatibilityError,IIDDistribution,trueD=measure().lattice())

    def test_IIDGen_in_QuasiClass(self):
        self.assertRaises(DistributionCompatibilityError,QuasiRandom,trueD=measure().stdGaussian())

if __name__ == "__main__":
    unittest.main()
