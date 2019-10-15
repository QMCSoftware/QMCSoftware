import unittest

from qmcpy._util import DistributionCompatibilityError
from qmcpy.discrete_distribution import IIDStdGaussian, Lattice
from qmcpy.stopping_criterion import CLT, CLTRep
from qmcpy.true_measure import Gaussian


class Test_CLT(unittest.TestCase):

    def test_raise_DistributionCompatibilityError(self):
        self.assertRaises(DistributionCompatibilityError, CLT, Lattice(),
                          Gaussian(3))


class Test_CLTRep(unittest.TestCase):

    def test_raise_DistributionCompatibilityError(self):
        self.assertRaises(DistributionCompatibilityError, CLTRep,
                          IIDStdGaussian(), Gaussian(3))


if __name__ == "__main__":
    unittest.main()
