""" Unit tests for discrete distributions in QMCPy """

from qmcpy import *

import unittest
from numpy import array, int64, log, ndarray, zeros
import numpy.testing as npt

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
        from qmcpy.discrete_distribution.mps_refactor import LatticeSeq
        from third_party.magic_point_shop import latticeseq_b2
        n, m = 4, 4
        gen_original_mps = latticeseq_b2(m=int(log(n) / log(2)), s=m)
        gen_qmcpy_mps = LatticeSeq(m=int(log(n) / log(2)), s=m, returnDeepCopy=False)
        true_array = array([
                [0,         0,          0,          0],
                [1 / 2,     1 / 2,      1 / 2,      1 / 2],
                [1 / 4,     3 / 4,      3 / 4,      1 / 4],
                [3 / 4,     1 / 4,      1 / 4,      3 / 4]])
        for gen in [gen_original_mps,gen_qmcpy_mps]:
            samples_unshifted = array([next(gen) for i in range(n)])
            self.assertTrue((samples_unshifted.squeeze() == true_array).all())


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
        from qmcpy.discrete_distribution.mps_refactor import DigitalSeq
        from third_party.magic_point_shop import digitalseq_b2g
        n, m = 4, 4
        gen_original_mps = digitalseq_b2g(
            Cs = "./third_party/magic_point_shop/sobol_Cs.col",
            m = int(log(n) / log(2)),
            s = m)
        gen_qmcpy_mps = DigitalSeq(
            Cs = "sobol_Cs.col",
            m = int(log(n) / log(2)),
            s = m,
            returnDeepCopy=False)
        true_array = array([
                [0,          0,          0,          0],
                [2147483648, 2147483648, 2147483648, 2147483648],
                [3221225472, 1073741824, 1073741824, 1073741824],
                [1073741824, 3221225472, 3221225472, 3221225472]])
        for gen in [gen_original_mps,gen_qmcpy_mps]:
            samples_unshifted = zeros((n, m), dtype=int64)
            for i, _ in enumerate(gen):
                samples_unshifted[i, :] = gen.cur
            self.assertTrue((samples_unshifted.squeeze() == true_array).all())

if __name__ == "__main__":
    unittest.main()
