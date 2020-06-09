""" Unit tests for discrete distributions in QMCPy """

from qmcpy import *
from qmcpy.util import *
from qmcpy.discrete_distribution.qrng.qrng import qrng_example_use
from numpy import *
import unittest


class TestIIDStdUniform(unittest.TestCase):
    """ Unit tests for IIDStdUniform DiscreteDistribution. """

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
    
    def test_set_dimension(self):
        distribution = IIDStdUniform(dimension=2)
        distribution.set_dimension(3)
        samples = distribution.gen_samples(4)
        self.assertTrue(samples.shape==(4,3))


class TestIIDGaussian(unittest.TestCase):
    """ Unit tests for IIDStdGaussian DiscreteDistribution. """

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

    def test_set_dimension(self):
        distribution = IIDStdGaussian(dimension=2)
        distribution.set_dimension(3)
        samples = distribution.gen_samples(4)
        self.assertTrue(samples.shape==(4,3))


class TestQRNG(unittest.TestCase):
    """ Unit tests for QRNG code from C """

    def test_qrng_example(self):
        qrng_example_use()


class TestLattice(unittest.TestCase):
    """ Unit tests for Lattice DiscreteDistribution. """

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
    
    def test_set_dimension(self):
        for backend in ['MPS','GAIL']:
            distribution = Lattice(dimension=2,backend=backend)
            distribution.set_dimension(3)
            samples = distribution.gen_samples(4)
            with self.subTest():
                self.assertTrue(samples.shape==(4,3))


class TestSobol(unittest.TestCase):
    """ Unit tests for Sobol DiscreteDistribution. """

    def test_mimics(self):
        distribution = Sobol(dimension=3, scramble=True, backend='QRNG')
        self.assertEqual(distribution.mimics, "StdUniform")

    def test_gen_samples(self):
        for backend in ['QRNG','MPS']:
            distribution = Sobol(dimension=3, scramble=True, backend=backend)
            samples = distribution.gen_samples(n_min=4, n_max=8)
            with self.subTest():
                self.assertEqual(type(samples), ndarray)
            with self.subTest():
                self.assertEqual(samples.shape, (4,3))

    def test_set_dimension(self):
        for backend in ['MPS','QRNG']:
            distribution = Sobol(dimension=2,backend=backend)
            distribution.set_dimension(3)
            samples = distribution.gen_samples(4)
            with self.subTest():
                self.assertTrue(samples.shape==(4,3))
    
    def test_qrng_graycode_ordering(self):
        s = Sobol(2,scramble=False,backend='qrng',graycode=True)
        x = s.gen_samples(n_min=4,n_max=8)
        x_true = array([
            [ 0.375,  0.375],
            [ 0.875,  0.875],
            [ 0.625,  0.125],
            [ 0.125,  0.625]])
        self.assertTrue((x==x_true).all())

    def test_qrng_natural_ordering(self):
        s = Sobol(2,scramble=False,backend='qrng',graycode=False)
        x = s.gen_samples(n_min=4,n_max=8)
        x_true = array([
            [ 0.125,  0.625],
            [ 0.625,  0.125],
            [ 0.375,  0.375],
            [ 0.875,  0.875]])
        self.assertTrue((x==x_true).all())


class TestCustomIIDDistribution(unittest.TestCase):
    """ Unit tests for CustomIIDDistribution DiscreteDistribution. """

    def test_gen_samples(self):
        distribution = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,2)))
        distribution.gen_samples(10)

    def test_set_dimension(self):
        distribution = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,2)))
        self.assertRaises(DimensionError,distribution.set_dimension,3)
                

class TestAcceptanceRejectionSampling(unittest.TestCase):
    """ Unit tests for AcceptanceRejectionSampling DiscreteDistribution. """

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
    
    def test_set_dimension(self):
        distribution = AcceptanceRejectionSampling(lambda x: 1, Uniform(IIDStdGaussian(2)))
        self.assertRaises(DimensionError,distribution.set_dimension,3)   


class TestInverseCDFSampling(unittest.TestCase):
    """ Unit tests for InverseCDFSampling DiscreteDistribution. """

    def test_gen_samples(self):
        distribution = InverseCDFSampling(Lattice(2),
            inverse_cdf_fun = lambda u,l=5: -log(1-u)/l)
                        # see sampling measures demo
        distribution.gen_samples(8)
    
    def test_set_dimension(self):
        distribution = InverseCDFSampling(Lattice(2),lambda u: u)
        self.assertRaises(DimensionError,distribution.set_dimension,3)   


if __name__ == "__main__":
    unittest.main()
