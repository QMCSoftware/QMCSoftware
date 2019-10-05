import unittest

from qmcpy.stop import DistributionCompatibilityError
from qmcpy.stop.clt_rep import CLTRep
from qmcpy.distribution.iid_distribution import IIDDistribution
from qmcpy.measures.measures import StdUniform

class Test_CLTRep(unittest.TestCase):

    def test_Incompatible_Distrib(self):
        self.assertRaises(DistributionCompatibilityError,CLTRep,IIDDistribution(StdUniform([2])))

    def test_max_samples_warning(self):
        pass

if __name__ == "__main__":
    unittest.main()
