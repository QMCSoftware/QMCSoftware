import unittest

from qmcpy.stop import CLT
from qmcpy.distribution import QuasiRandom
from qmcpy.measures import Lattice
from qmcpy._util import DistributionCompatibilityError

class Test_CLT(unittest.TestCase):

    def test_Incompatible_Distrib(self):
        self.assertRaises(DistributionCompatibilityError,CLT,QuasiRandom(Lattice([2])))

    def test_max_samples_warning(self):
        pass

if __name__ == "__main__":
    unittest.main()
