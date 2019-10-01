import unittest

from algorithms.stop import DistributionCompatibilityError
from algorithms.stop.CLTStopping import CLTStopping
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.distribution.Measures import Lattice

class Test_CLTStopping(unittest.TestCase):

    def test_Incompatible_Distrib(self):
        self.assertRaises(DistributionCompatibilityError,CLTStopping,QuasiRandom(Lattice([2])))
    
    def test_max_samples_warning(self):
        pass

if __name__ == "__main__":
    unittest.main()
