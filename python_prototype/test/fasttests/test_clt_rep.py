import unittest

from qmcpy.stop import CLTRep
from qmcpy.distribution import IIDDistribution
from qmcpy.measures import StdUniform
from qmcpy._util import DistributionCompatibilityError


class Test_CLTRep(unittest.TestCase):
    def test_Incompatible_Distrib(self):
        self.assertRaises(DistributionCompatibilityError, CLTRep,
                          IIDDistribution(StdUniform([2])))


    def test_max_samples_warning(self):
        pass


if __name__ == "__main__":
    unittest.main()
