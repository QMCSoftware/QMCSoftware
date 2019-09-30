import unittest

from algorithms.distribution import Measure
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution import MeasureCompatibilityError


class Test_IIDDistribution(unittest.TestCase):

    def test_QuasiGen_in_IIDClass(self):
        self.assertRaises(MeasureCompatibilityError,IIDDistribution,trueD=Measure().lattice())

if __name__ == "__main__":
    unittest.main()