import unittest

from algorithms.measures.measures import Lattice
from algorithms.distribution.iid_distribution import IIDDistribution
from algorithms.distribution import MeasureCompatibilityError

class Test_IIDDistribution(unittest.TestCase):

    def test_QuasiGen_in_IIDClass(self):
        self.assertRaises(MeasureCompatibilityError,IIDDistribution,true_distribution=Lattice([2]))

if __name__ == "__main__":
    unittest.main()