""" Unit tests for discrete distributions in QMCPy """

from qmcpy import *
from numpy import array, int64, log2, ndarray, vstack, zeros, random, log
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
        distribution = Lattice(dimension=3, scramble=True, backend='MPS')
        self.assertEqual(distribution.mimics, "StdUniform")

    def test_gen_samples(self):
        for backend in ['MPS','GAIL']:
            distribution = Lattice(dimension=3, scramble=True, backend=backend)
            samples = distribution.gen_samples(n_min=4, n_max=8)
            with self.subTest():
                self.assertEqual(type(samples), ndarray)
            with self.subTest():
                self.assertEqual(samples.shape, (4,3))

    def test_mps_correctness(self):
        distribution = Lattice(dimension=4, scramble=False, backend='MPS')
        true_sample = array([
            [0,     0,      0,      0],
            [1/2,   1/2,    1/2,    1/2],
            [1/4,   3/4,    3/4,    1/4],
            [3/4,   1/4,    1/4,    3/4]])
        self.assertTrue((distribution.gen_samples(n_min=0,n_max=4)==true_sample).all())

    def test_gail_correctness(self):
        distribution = Lattice(dimension=4, scramble=False, backend='GAIL')
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
        distribution = Sobol(dimension=3, scramble=True, backend='QRNG')
        self.assertEqual(distribution.mimics, "StdUniform")

    def test_gen_samples(self):
        for backend in ['QRNG','MPS','PyTorch']:
            distribution = Sobol(dimension=3, scramble=True, backend=backend)
            samples = distribution.gen_samples(n_min=4, n_max=8)
            with self.subTest():
                self.assertEqual(type(samples), ndarray)
            with self.subTest():
                self.assertEqual(samples.shape, (4,3))


class TestCustomIIDDistribution(unittest.TestCase):
    """
    Unit tests for CustomIIDDistribution
    """

    def test_gen_samples(self):
        distribution = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,2)))
        distribution.gen_samples(10)


class TestAcceptanceRejectionSampling(unittest.TestCase):
    """
    Unit tests for AcceptanceRejectionSampling
    """

    def test_gen_samples(self):
        def f(x):
            # see sampling measures demo
            x = x if x<.5 else 1-x 
            density = 16*x/3 if x<1/4 else 4/3
            return density  
        distribution = AcceptanceRejectionSampling(
            objective_pdf = f,
            measure_to_sample_from = Uniform(IIDStdUniform(1)))
        distribution.gen_samples(10)


class TestInverseCDFSampling(unittest.TestCase):
    """
    Unit tests for InverseCDFSampling
    """

    def test_gen_samples(self):
        distribution = InverseCDFSampling(Lattice(2),
            inverse_cdf_fun = lambda u,l=5: -log(1-u)/l)
                        # see sampling measures demo
        distribution.gen_samples(8)


if __name__ == "__main__":
    unittest.main()
