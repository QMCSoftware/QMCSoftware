import unittest

from algorithms.distribution.Measures import Lattice
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution import MeasureCompatibilityError

class Test_IIDDistribution(unittest.TestCase):

    def test_QuasiGen_in_IIDClass(self):
        self.assertRaises(MeasureCompatibilityError,IIDDistribution,true_distrib=Lattice([2]))

if __name__ == "__main__":
    unittest.main()