import unittest

from qmcpy._util import DistributionCompatibilityError
from qmcpy.discrete_distribution import IIDStdGaussian, Lattice
from qmcpy.stopping_criterion import CLT, CLTRep
from qmcpy.true_measure import Gaussian


class TestClt(unittest.TestCase):

    def test_raise_distribution_compatibility_error(self):
        self.assertRaises(DistributionCompatibilityError, CLT, Lattice(),
                          Gaussian(3))


class TestCltRep(unittest.TestCase):

    def test_raise_distribution_compatibility_error(self):
        self.assertRaises(DistributionCompatibilityError, CLTRep,
                          IIDStdGaussian(), Gaussian(3))


if __name__ == "__main__":
    unittest.main()
