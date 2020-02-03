""" Unit tests for subclasses of TrueMeasures in QMCPy """

import unittest

from numpy import array
from qmcpy import *
from qmcpy.util import *


class TestTrueDistributionConstruction(unittest.TestCase):
    """
    Unit tests for Gaussian distribution in QMCPy.
    """

    def test_dimensions(self):
        measure = Gaussian(1)
        self.assertTrue(all(measure.dimension == array([1])))
        measure = Gaussian([1, 2, 3])
        self.assertTrue(all(measure.dimension == array([1, 2, 3])))
        self.assertRaises(DimensionError, Gaussian, [1, 1.5, 2])

    def test_distribute_attributes(self):
        measure = Gaussian([1, 2, 3], mean=[-2, 0, 2])
        self.assertEqual(measure[0].sigma, 1)
        self.assertEqual(measure[2].mu, 2)

    def test_transform_errors(self):
        distrib = IIDStdGaussian()
        distrib.mimics = "Poisson"
        # Transform from poisson to Gaussion not implemented yet
        measure = Gaussian(2)
        self.assertRaises(TransformError, measure.set_tm_gen,
                          distrib)
        measure.set_tm_gen(IIDStdGaussian())
        self.assertRaises(TransformError, measure.gen_tm_samples, 3, 5)

    def test_mismatched_dimensions(self):
        measure = Uniform(1)
        integrand = Keister(2)
        self.assertRaises(DimensionError, integrate, integrand, measure)
        measure = Uniform([1, 2, 3])
        integrand = Keister([1, 2, 4])
        self.assertRaises(DimensionError, integrate, integrand, measure)


class TestUniform(unittest.TestCase):
    """
    Unit tests for Uniform distribution in QMCPy.
    """

    def test_transforms(self):
        a, b = 1 / 4, 1 / 2
        measure = Uniform(3, lower_bound=a, upper_bound=b)
        measure.set_tm_gen(IIDStdUniform())
        # IIDStdUniform -> Uniform(1/4,1/2)
        vals = measure[0].gen_tm_samples(5, 4)
        self.assertTrue((vals <= b).all() and (vals >= a).all())
        measure = Uniform(3, lower_bound=a, upper_bound=b)
        measure.set_tm_gen(IIDStdGaussian())
        # IIDStdGaussian -> Uniform(1/4,1/2)
        vals = measure[0].gen_tm_samples(5, 4)
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
