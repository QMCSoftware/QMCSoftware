import unittest

from algorithms.stop import DistributionCompatibilityError
from algorithms.stop.CLT_Rep import CLT_Rep
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution import measure

class Test_CLT_Rep(unittest.TestCase):

    def test_Incompatible_Distrib(self):
        self.assertRaises(DistributionCompatibilityError,CLT_Rep,IIDDistribution(measure().stdUniform()))

if __name__ == "__main__":
    unittest.main()
