import unittest

from numpy import array, int64, log, zeros
from qmcpy._util import MeasureCompatibilityError
from qmcpy.distribution import QuasiRandom
from qmcpy.third_party.magic_point_shop import LatticeSeq
from qmcpy.measures import StdGaussian

from qmcpy.distribution.digital_seq import DigitalSeq


class Test_IIDDistribution(unittest.TestCase):
    def test_IIDGen_in_QuasiClass(self):
        self.assertRaises(MeasureCompatibilityError, QuasiRandom,
                          true_distribution=StdGaussian([2]))
    def test_backend_lattice(self):
        n, m = 4, 4
        array_not_shifted = array([LatticeSeq(m=int(log(n) / log(2)), s=m)])
        true_array = array([[0,   0,   0,   0  ],
                            [1/2, 1/2, 1/2, 1/2],
                            [1/4, 3/4, 3/4, 1/4],
                            [3/4, 1/4, 1/4, 3/4]])
        array_not_shifted = array([row for row in
                                   LatticeSeq(m=int(log(n)/log(2)), s=m)])

    def test_backend_sobol(self):
        n, m = 4, 4
        gen = DigitalSeq(Cs="sobol_Cs.col", m=int(log(n) / log(2)), s=m)
        array_not_shifted = zeros((n, m), dtype=int64)
        for i, row in enumerate(gen):
            array_not_shifted[i, :] = gen.cur  # set each nxm
        true_array = array([[0,          0,          0,          0         ],
                            [2147483648, 2147483648, 2147483648, 2147483648],
                            [3221225472, 1073741824, 1073741824, 1073741824],
                            [1073741824, 3221225472, 3221225472, 3221225472]])
        self.assertTrue((array_not_shifted.squeeze() == true_array).all())


if __name__ == "__main__":
    unittest.main()
