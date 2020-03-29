""" Unit tests for discrete distributions in QMCPy """

from qmcpy import *
from numpy import array, int64, log2, ndarray, vstack, zeros
import unittest


class TestIIDStdUniform(unittest.TestCase):
    """
    Unit tests for IIDStdUniform in QMCPy.
    """

    def test_mimics(self):
        distribution = IIDStdUniform(dimension=3)
        self.assertEqual(distribution.mimics, "StdUniform")

    def test_gen_samples(self):
        distribution = IIDStdUniform(dimension=3)
        samples = distribution.gen_samples(n=5)
        with self.subTest():
            self.assertEqual(type(samples), ndarray)
        with self.subTest():
            self.assertEqual(samples.shape, (5,3))


class TestIIDGaussian(unittest.TestCase):
    """
    Unit tests for IIDStdGaussian in QMCPy.
    """

    def test_mimics(self):
        distribution = IIDStdGaussian(dimension=3)
        self.assertEqual(distribution.mimics, "StdGaussian")

    def test_gen_samples(self):
        distribution = IIDStdGaussian(dimension=3)
        samples = distribution.gen_samples(n=5)
        with self.subTest():
            self.assertEqual(type(samples), ndarray)
        with self.subTest():
            self.assertEqual(samples.shape, (5,3))


class TestLattice(unittest.TestCase):
    """
    Unit tests for Lattice sampling points in QMCPy.
    """

    def test_mimics(self):
        distribution = Lattice(dimension=3, replications=2, scramble=True, backend='MPS')
        self.assertEqual(distribution.mimics, "StdUniform")

    def test_gen_samples(self):
        distribution1 = Lattice(dimension=3, replications=2, scramble=True, backend='MPS')
        distribution2 = Lattice(dimension=3, replications=2, scramble=True, backend='GAIL')
        for distribution in [distribution1,distribution2]:
            samples = distribution.gen_samples(n_min=4, n_max=8)
            with self.subTest():
                self.assertEqual(type(samples), ndarray)
            with self.subTest():
                self.assertEqual(samples.shape, (2,4,3))

    def test_mps_correctness(self):
        distribution = Lattice(dimension=4, replications=0, scramble=False, backend='MPS')
        true_sample = array([
            [0,     0,      0,      0],
            [1/2,   1/2,    1/2,    1/2],
            [1/4,   3/4,    3/4,    1/4],
            [3/4,   1/4,    1/4,    3/4]])
        self.assertTrue((distribution.gen_samples(n_min=0,n_max=4)==true_sample).all())

    def test_gail_correctness(self):
        distribution = Lattice(dimension=4, replications=0, scramble=False, backend='GAIL')
        true_sample = array([
            [0,     0,      0,      0],
            [1/2,   1/2,    1/2,    1/2],
            [1/4,   1/4,    1/4,    1/4],
            [3/4,   3/4,    3/4,    3/4]])
        self.assertTrue((distribution.gen_samples(n_min=0,n_max=4)==true_sample).all())


class TestSobol(unittest.TestCase):
    """
    Unit tests for Sobol sampling points in QMCPy.
    """

    def test_mimics(self):
        distribution = Sobol(dimension=3, replications=2, scramble=True, backend='MPS')
        self.assertEqual(distribution.mimics, "StdUniform")

    def test_gen_samples(self):
        distribution1 = Sobol(dimension=3, replications=2, scramble=True, backend='MPS')
        distribution2 = Sobol(dimension=3, replications=2, scramble=True, backend='PyTorch')
        for distribution in [distribution1,distribution2]:
            samples = distribution.gen_samples(n_min=4, n_max=8)
            with self.subTest():
                self.assertEqual(type(samples), ndarray)
            with self.subTest():
                self.assertEqual(samples.shape, (2,4,3))

if __name__ == "__main__":
    unittest.main()
