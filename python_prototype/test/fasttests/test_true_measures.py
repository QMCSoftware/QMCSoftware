""" Unit tests for subclasses of TrueMeasures in QMCPy """

from qmcpy import *
from qmcpy._util import *

import unittest


class TestTrueDistributionConstruction(unittest.TestCase):
    """
    Unit tests for Gaussian distribution in QMCPy.
    """

    def test_dimensions(self):
        true_measure = Gaussian(1)
        self.assertTrue(all(true_measure.dimension == array([1])))
        true_measure = Gaussian([1, 2, 3])
        self.assertTrue(all(true_measure.dimension == array([1, 2, 3])))
        self.assertRaises(DimensionError, Gaussian, [1, 1.5, 2])

    def test_distribute_attributes(self):
        true_measure = Gaussian([1, 2, 3], mean=[-2, 0, 2])
        self.assertEqual(true_measure[0].sigma, 1)
        self.assertEqual(true_measure[2].mu, 2)
        
    def test_transform_errors(self):
        discrete_distrib = IIDStdGaussian()
        discrete_distrib.mimics = "Poisson"
        # Transform from poisson to Gaussion not implemented yet
        true_measure = Gaussian(2)
        self.assertRaises(TransformError, true_measure.set_tm_gen,
                          discrete_distrib)
        true_measure.set_tm_gen(IIDStdGaussian())
        self.assertRaises(TransformError, true_measure.gen_tm_samples, 3, 5)


class TestUniform(unittest.TestCase):
    """
    Unit tests for Uniform distribution in QMCPy.
    """

    def test_transforms(self):
        a, b = 1 / 4, 1 / 2
        true_measure = Uniform(3, lower_bound=a, upper_bound=b)
        true_measure.set_tm_gen(IIDStdUniform())
        # IIDStdUniform -> Uniform(1/4,1/2)
        vals = true_measure[0].gen_tm_samples(5, 4)
        self.assertTrue((vals <= b).all() and (vals >= a).all())
        true_measure = Uniform(3, lower_bound=a, upper_bound=b)
        true_measure.set_tm_gen(IIDStdGaussian())
        # IIDStdGaussian -> Uniform(1/4,1/2)
        vals = true_measure[0].gen_tm_samples(5, 4)
        self.assertTrue((vals <= b).all() and (vals >= a).all())


class TestGaussian(unittest.TestCase):
    """
    Unit tests for Gaussian distribution in QMCPy.
    """

    def test_transforms(self):
        pass


class TestBrownianMontion(unittest.TestCase):
    """
    Unit tests for Brownian Motion in QMCPy.
    """

    def test_transforms(self):
        pass


if __name__ == "__main__":
    unittest.main()
