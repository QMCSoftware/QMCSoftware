import unittest
from numpy import array,log,zeros,int64

from qmcpy.distribution.magic_point_shop.latticeseq_b2 import latticeseq_b2
from qmcpy.measures.measures import StdGaussian
from qmcpy.distribution.quasi_random import QuasiRandom
from qmcpy.distribution import MeasureCompatibilityError
from qmcpy.distribution.magic_point_shop.digital_seq import DigitalSeq


class Test_IIDDistribution(unittest.TestCase):

    def test_IIDGen_in_QuasiClass(self):
        self.assertRaises(MeasureCompatibilityError,QuasiRandom,true_distribution=StdGaussian([2]))

    def test_backend_lattice(self):
        n,m = 4,4
        array_not_shifted = array([row for row in latticeseq_b2(m=int(log(n)/log(2)), s=m)])
        true_array = array([[0,   0,   0,   0  ],
                            [1/2, 1/2, 1/2, 1/2],
                            [1/4, 3/4, 3/4, 1/4],
                            [3/4, 1/4, 1/4, 3/4]])
        self.assertTrue((array_not_shifted.squeeze()==true_array).all())

    def test_backend_sobol(self):
        n,m = 4,4
        gen = DigitalSeq(Cs='sobol_Cs.col', m=int(log(n)/log(2)),s=m)
        array_not_shifted = zeros((n,m),dtype=int64)
        for i,row in enumerate(gen):
            array_not_shifted[i,:] = gen.cur # set each nxm
        true_array = array([[0,          0,          0,          0         ],
                            [2147483648, 2147483648, 2147483648, 2147483648],
                            [3221225472, 1073741824, 1073741824, 1073741824],
                            [1073741824, 3221225472, 3221225472, 3221225472]])
        self.assertTrue((array_not_shifted.squeeze()==true_array).all())


if __name__ == "__main__":
    unittest.main()