import unittest

from numpy import array, int64, log, ndarray, zeros
from qmcpy.discrete_distribution import *
from qmcpy.discrete_distribution.digital_seq import DigitalSeq
from qmcpy.third_party.magic_point_shop import LatticeSeq


class Test_IIDStdUniform(unittest.TestCase):

    def test_mimics(self):
        discrete_distrib = IIDStdUniform()
        self.assertEqual(discrete_distrib.mimics, 'StdUniform')

    def test_gen_samples(self):
        discrete_distrib = IIDStdUniform()
        samples = discrete_distrib.gen_dd_samples(1, 2, 3)
        with self.subTest():
            self.assertEqual(type(samples), ndarray)
        with self.subTest():
            self.assertEqual(samples.shape, (1, 2, 3))


class Test_IIDGaussian(unittest.TestCase):

    def test_mimics(self):
        discrete_distrib = IIDStdGaussian()
        self.assertEqual(discrete_distrib.mimics, 'StdGaussian')

    def test_gen_samples(self):
        discrete_distrib = IIDStdGaussian()
        samples = discrete_distrib.gen_dd_samples(1, 2, 3)
        with self.subTest():
            self.assertEqual(type(samples), ndarray)
        with self.subTest():
            self.assertEqual(samples.shape, (1, 2, 3))


class Test_Lattice(unittest.TestCase):

    def test_mimics(self):
        discrete_distrib = Lattice()
        self.assertEqual(discrete_distrib.mimics, 'StdUniform')

    def test_gen_samples(self):
        discrete_distrib = Lattice()
        samples = discrete_distrib.gen_dd_samples(1, 2, 3)
        with self.subTest():
            self.assertEqual(type(samples), ndarray)
        with self.subTest():
            self.assertEqual(samples.shape, (1, 2, 3))

    def test_backend_lattice(self):
        n, m = 4, 4
        array_not_shifted = array(
            [row for row in LatticeSeq(m=int(log(n) / log(2)), s=m)])
        true_array = array([[0, 0, 0, 0],
                            [1 / 2, 1 / 2, 1 / 2, 1 / 2],
                            [1 / 4, 3 / 4, 3 / 4, 1 / 4],
                            [3 / 4, 1 / 4, 1 / 4, 3 / 4]])
        self.assertTrue((array_not_shifted.squeeze() == true_array).all())


class Test_Sobol(unittest.TestCase):

    def test_mimics(self):
        discrete_distrib = Sobol()
        self.assertEqual(discrete_distrib.mimics, 'StdUniform')

    def test_gen_samples(self):
        discrete_distrib = Sobol()
        samples = discrete_distrib.gen_dd_samples(1, 2, 3)
        with self.subTest():
            self.assertEqual(type(samples), ndarray)
        with self.subTest():
            self.assertEqual(samples.shape, (1, 2, 3))

    def test_backend_sobol(self):
        n, m = 4, 4
        gen = DigitalSeq(Cs="sobol_Cs.col", m=int(log(n) / log(2)), s=m)
        array_not_shifted = zeros((n, m), dtype=int64)
        for i, row in enumerate(gen):
            array_not_shifted[i, :] = gen.cur  # set each nxm
        true_array = array([[0, 0, 0, 0],
                            [2147483648, 2147483648, 2147483648, 2147483648],
                            [3221225472, 1073741824, 1073741824, 1073741824],
                            [1073741824, 3221225472, 3221225472, 3221225472]])
        self.assertTrue((array_not_shifted.squeeze() == true_array).all())


if __name__ == "__main__":
    unittest.main()
