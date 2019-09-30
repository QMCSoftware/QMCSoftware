import unittest

from algorithms.stop import DistributionCompatibilityError
from algorithms.stop.CLTRep import CLTRep
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution import Measure

class Test_CLTRep(unittest.TestCase):

    def test_Incompatible_Distrib(self):
        self.assertRaises(DistributionCompatibilityError,CLTRep,IIDDistribution(Measure().std_uniform()))

if __name__ == "__main__":
    unittest.main()
