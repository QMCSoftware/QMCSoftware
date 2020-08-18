""" Unit tests for discrete distributions in QMCPy """

from qmcpy import *
from qmcpy.util import *
from qmcpy.util import ParameterError,ParameterWarning
from numpy import *
import sys
vinvo = sys.version_info
if vinvo[0]==3: import unittest
else: import unittest2 as unittest


class TestIIDStdUniform(unittest.TestCase):
    """ Unit tests for IIDStdUniform DiscreteDistribution. """

    def test_mimics(self):
        distribution = IIDStdUniform(dimension=3)
        self.assertEqual(distribution.mimics, "StdUniform")

    def test_gen_samples(self):
        distribution = IIDStdUniform(dimension=3)
        samples = distribution.gen_samples(n=5)
        self.assertEqual(type(samples), ndarray)
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
        self.assertEqual(type(samples), ndarray)
        self.assertEqual(samples.shape, (5,3))

    def test_set_dimension(self):
        distribution = IIDStdGaussian(dimension=2)
        distribution.set_dimension(3)
        samples = distribution.gen_samples(4)
        self.assertTrue(samples.shape==(4,3))


class TestLattice(unittest.TestCase):
    """ Unit tests for Lattice DiscreteDistribution. """

    def test_mimics(self):
        distribution = Lattice(dimension=3, randomize=True, backend='MPS')
        self.assertEqual(distribution.mimics, "StdUniform")

    def test_gen_samples(self):
        for backend in ['MPS','GAIL']:
            distribution = Lattice(dimension=3, randomize=True, backend=backend)
            samples = distribution.gen_samples(n_min=4, n_max=8)
            self.assertEqual(type(samples), ndarray)
            self.assertEqual(samples.shape, (4,3))

    def test_mps_correctness(self):
        distribution = Lattice(dimension=4, randomize=False, backend='MPS')
        true_sample = array([
            [1./8,   3./8,    3./8,    1./8],
            [3./8,   1./8,    1./8,    3./8],
            [5./8,   7./8,    7./8,    5./8],
            [7./8,   5./8,    5./8,    7./8]])
        self.assertTrue((distribution.gen_samples(n_min=4,n_max=8)==true_sample).all())

    def test_gail_correctness(self):
        distribution = Lattice(dimension=4, randomize=False, backend='GAIL')
        true_sample = array([
            [1./8,   3./8,    3./8,    1./8],
            [5./8,   7./8,    7./8,    5./8],
            [3./8,   1./8,    1./8,    3./8],
            [7./8,   5./8,    5./8,    7./8]])
        self.assertTrue((distribution.gen_samples(n_min=4,n_max=8)==true_sample).all())
    
    def test_set_dimension(self):
        for backend in ['MPS','GAIL']:
            distribution = Lattice(dimension=2,backend=backend)
            distribution.set_dimension(3)
            samples = distribution.gen_samples(4)
            self.assertTrue(samples.shape==(4,3))
    
    def test_custom_vector(self):
        self.assertRaises(ParameterError,Lattice,dimension=4,gen_vector_info={'vector':[1,433461,315689],'n_lim':2**20})
        l = Lattice(2,gen_vector_info={'vector':[1,433461,315689],'n_lim':2**20})
        self.assertRaises(ParameterError,l.gen_samples,n_min=4,n_max=2**21)
        l.gen_samples(n_min=3,n_max=5)

class TestSobol(unittest.TestCase):
    """ Unit tests for Sobol DiscreteDistribution. """

    def test_mimics(self):
        distribution = Sobol(dimension=3, randomize=True, backend='QRNG')
        self.assertEqual(distribution.mimics, "StdUniform")

    def test_gen_samples_and_set_dimension(self):
        dds = [
            Sobol(dimension=2, randomize=True, backend='QRNG', graycode=True),
            Sobol(dimension=2, randomize=True, backend='PyTorch', graycode=True),
            Sobol(dimension=2, randomize=False, backend='Seq51', graycode=False)]
        for dd in dds:
            samples = dd.gen_samples(n_min=4, n_max=8)
            self.assertEqual(type(samples), ndarray)
            self.assertEqual(samples.shape, (4,2))
            dd.set_dimension(3)
            samples = dd.gen_samples(4)
            self.assertTrue(samples.shape==(4,3))
    
    def test_qrng_graycode_ordering(self):
        s = Sobol(2,randomize=False,backend='qrng',graycode=True)
        x = s.gen_samples(n_min=4,n_max=8)
        x_true = array([
            [ 0.375,  0.375],
            [ 0.875,  0.875],
            [ 0.625,  0.125],
            [ 0.125,  0.625]])
        self.assertTrue((x==x_true).all())

    def test_qrng_natural_ordering(self):
        s = Sobol(2,randomize=False,backend='qrng',graycode=False)
        x = s.gen_samples(n_min=4,n_max=8)
        x_true = array([
            [ 0.125,  0.625],
            [ 0.625,  0.125],
            [ 0.375,  0.375],
            [ 0.875,  0.875]])
        self.assertTrue((x==x_true).all())
    
    def test_pytorch_0th_vector(self):
        x = Sobol(1,randomize=False,backend='PyTorch',graycode=True).gen_samples(4)
        self.assertTrue((x==array([[1./2,3./4,1./4,3./8]]).T).all())

class TestHalton(unittest.TestCase):
    """ Unit test for Halton DiscreteDistribution. """

    def test_mimics(self):
        distribution = Halton(dimension=3, generalize=True, seed=7)
        self.assertEqual(distribution.mimics, "StdUniform")
    
    def test_gen_samples(self):
        distribution = Halton(dimension=3, generalize=True, seed=7)
        x = distribution.gen_samples(4)
        self.assertTrue(x.shape==(4,3))
        distribution.set_dimension(4)
        x = distribution.gen_samples(2)
        self.assertTrue(x.shape==(2,4))
        distribution = Halton(dimension=3, generalize=True, backend='Owen')
        x = distribution.gen_samples(4)
        self.assertTrue(x.shape==(4,3))
        distribution.set_dimension(4)
        x = distribution.gen_samples(3)
        self.assertTrue(x.shape==(3,4))
        x2 = distribution.gen_samples(n_min=1,n_max=3)
        self.assertTrue(x2.shape==(2,4))
        self.assertTrue((x2==x[1:]).all())
        x_true = array([
            [0,     0],
            [1./2,  1./3],
            [1./4,  2./3],
            [3./4,  1./9]])
        x = Halton(2,randomize=False,backend='Owen',generalize=True).gen_samples(4)
        self.assertTrue((x==x_true).all())
    
    def test_warnings_errors(self):
        self.assertRaises(ParameterError,Halton,2,generalize=False,backend='Owen')
        self.assertRaises(ParameterError,Halton,2,randomize=False,generalize=False,backend='QRNG')
        distribution = Halton(2, generalize=True, backend='QRNG',seed=7)
        self.assertRaises(ParameterError,distribution.gen_samples,n_min=2,n_max=4)

class TestKorobov(unittest.TestCase):
    """ Unit test for Korobov DiscreteDistribution. """

    def test_mimics(self):
        distribution = Korobov(dimension=3, generator=[1], randomize=True, seed=None)
        self.assertEqual(distribution.mimics, "StdUniform")
    
    def test_gen_samples(self):
        distribution = Korobov(dimension=3,generator=[1,2,3],randomize=False,seed=7)
        x = distribution.gen_samples(4)
        self.assertTrue(x.shape==(4,3))
        self.assertRaises(ParameterError,distribution.gen_samples,3)
        distribution = Korobov(dimension=2,generator=[1,3],randomize=False,seed=7)
        x = distribution.gen_samples(4)
        x_true = array([
            [0,     0],
            [1./4,  3./4],
            [1./2,  1./2],
            [3./4,  1./4]])
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
