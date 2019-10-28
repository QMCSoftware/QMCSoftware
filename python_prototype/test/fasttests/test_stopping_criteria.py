import unittest
from numpy import array

from qmcpy._util import DistributionCompatibilityError,MaxSamplesWarning
from qmcpy.discrete_distribution import IIDStdGaussian, Lattice
from qmcpy.stopping_criterion import CLT, CLTRep
from qmcpy.true_measure import Gaussian

class TestConstruction(unittest.TestCase):
    
    def test_check_n(self):
        stop_obj = CLT(IIDStdGaussian(),Gaussian(2))
        stop_obj.n_max = 70 # 70 samples max
        stop_obj.data.n_total = 50 # already generate 50 samples
        n_next = array([2,4,6,8,10]) # need to generate  30 more
        # need a net decrease of 10 samples before next generation
        with self.subTest(): # raise max_sample warning
            with self.assertWarns(MaxSamplesWarning):
                self.assertTrue(all(stop_obj.check_n(n_next)[1]==array([1,2,4,5,6])))            
        stop_obj.n_max = 100 # n_next is possible
        with self.subTest(): # no warnings. Returns itself
            self.assertTrue(stop_obj.check_n(n_next)[0]==0)

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