""" Unit tests for discrete distributions in QMCPy """

import unittest

from numpy import array, int64, log2, ndarray, vstack, zeros
from qmcpy import *
from qmcpy.discrete_distribution.mps_refactor import DigitalSeq, LatticeSeq
from third_party.magic_point_shop import digitalseq_b2g, latticeseq_b2


class TestIIDStdUniform(unittest.TestCase):
    """
    Unit tests for IIDStdUniform in QMCPy.
    """

    def test_mimics(self):
        discrete_distrib = IIDStdUniform()
        self.assertEqual(discrete_distrib.mimics, "StdUniform")

    def test_gen_samples(self):
        discrete_distrib = IIDStdUniform()
        samples = discrete_distrib.gen_dd_samples(1, 2, 3)
        with self.subTest():
            self.assertEqual(type(samples), ndarray)
        with self.subTest():
            self.assertEqual(samples.shape, (1, 2, 3))


class TestIIDGaussian(unittest.TestCase):
    """
    Unit tests for IIDStdGaussian in QMCPy.
    """

    def test_mimics(self):
        discrete_distrib = IIDStdGaussian()
        self.assertEqual(discrete_distrib.mimics, "StdGaussian")

    def test_gen_samples(self):
        discrete_distrib = IIDStdGaussian()
        samples = discrete_distrib.gen_dd_samples(1, 2, 3)
        with self.subTest():
            self.assertEqual(type(samples), ndarray)
        with self.subTest():
            self.assertEqual(samples.shape, (1, 2, 3))


class TestLattice(unittest.TestCase):
    """
    Unit tests for Lattice sampling points in QMCPy.
    """

    def test_mimics(self):
        discrete_distrib = Lattice()
        self.assertEqual(discrete_distrib.mimics, "StdUniform")

    def test_gen_samples(self):
        discrete_distrib = Lattice()
        samples = discrete_distrib.gen_dd_samples(1, 2, 3)
        with self.subTest():
            self.assertEqual(type(samples), ndarray)
        with self.subTest():
            self.assertEqual(samples.shape, (1, 2, 3))

    def test_backend_lattice(self):
        n, d = 4, 4
        true_samples = array([
            [0, 0, 0, 0],
            [1 / 2, 1 / 2, 1 / 2, 1 / 2],
            [1 / 4, 3 / 4, 3 / 4, 1 / 4],
            [3 / 4, 1 / 4, 1 / 4, 3 / 4]])
        # Original MPS Generator with standard method
        gen_original_mps = latticeseq_b2(s=d)
        mps_sampels = array([next(gen_original_mps) for i in range(n)])
        self.assertTrue(all(row in mps_sampels for row in true_samples))
        # QMCPy Generator with calc_block method (based on MPS implementation)
        qmcpy_gen = LatticeSeq(s=d)
        qmcpy_samples = vstack([qmcpy_gen.calc_block(m) for m in range(int(log2(n)) + 1)])
        self.assertTrue(all(row in qmcpy_samples for row in true_samples))


class TestSobol(unittest.TestCase):
    """
    Unit tests for Sobol sampling points in QMCPy.
    """

    def test_mimics(self):
        discrete_distrib = Sobol()
        self.assertEqual(discrete_distrib.mimics, "StdUniform")

    def test_gen_samples(self):
        discrete_distrib = Sobol()
        samples = discrete_distrib.gen_dd_samples(1, 2, 3)
        with self.subTest():
            self.assertEqual(type(samples), ndarray)
        with self.subTest():
            self.assertEqual(samples.shape, (1, 2, 3))

    def test_backend_sobol(self):
        n, m = 4, 4
        gen_original_mps = digitalseq_b2g(
            Cs="./third_party/magic_point_shop/sobol_Cs.col",
            m=int(log2(n)),
            s=m)
        gen_qmcpy_mps = DigitalSeq(
            Cs="sobol_Cs.col",
            m=int(log2(n)),
            s=m)
        true_array = array([
            [0, 0, 0, 0],
            [2147483648, 2147483648, 2147483648, 2147483648],
            [3221225472, 1073741824, 1073741824, 1073741824],
            [1073741824, 3221225472, 3221225472, 3221225472]])

        for gen in [gen_original_mps, gen_qmcpy_mps]:
            samples_unshifted = zeros((n, m), dtype=int64)
            for i, _ in enumerate(gen):
                samples_unshifted[i, :] = gen.cur
            self.assertTrue((samples_unshifted.squeeze() == true_array).all())

        del gen_qmcpy_mps # Ensure shallow copy d.n. affect samples_unshifted
        self.assertTrue((samples_unshifted == true_array).all())


if __name__ == "__main__":
    unittest.main()
