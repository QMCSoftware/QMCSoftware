import unittest

from qmcpy.stop import DistributionCompatibilityError
from qmcpy.stop.clt_stopping import CLTStopping
from qmcpy.distribution.quasi_random import QuasiRandom
from qmcpy.measures.measures import Lattice

class Test_CLTStopping(unittest.TestCase):

    def test_Incompatible_Distrib(self):
        self.assertRaises(DistributionCompatibilityError,CLTStopping,QuasiRandom(Lattice([2])))

    def test_max_samples_warning(self):
        pass

if __name__ == "__main__":
    unittest.main()
