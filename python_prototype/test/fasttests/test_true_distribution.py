import unittest
from numpy import array, arange

from qmcpy.discrete_distribution import IIDStdUniform,IIDStdGaussian
from qmcpy.true_distribution import Uniform,Gaussian,BrownianMotion
from qmcpy._util import DimensionError,TransformError

class Test_TrueDistribution_construciton(unittest.TestCase):
    
    def test_dimensions(self):
        true_distrib = Gaussian(1)
        with self.subTest(): self.assertTrue(all(true_distrib.dimension==array([1])))
        true_distrib = Gaussian([1,2,3])
        with self.subTest(): self.assertTrue(all(true_distrib.dimension==array([1,2,3])))
        with self.subTest(): self.assertRaises(DimensionError,Gaussian,[1,1.5,2])
    
    def test_distribute_attributes(self):
        true_distrib = Gaussian([1,2,3],mean=[-2,0,2])
        with self.subTest(): self.assertEqual(true_distrib[0].sigma,1)
        with self.subTest(): self.assertEqual(true_distrib[2].mu,2)
        with self.subTest(): self.assertRaises(DimensionError,Gaussian,[1,2,3],[-1,0])

    def test_transform(self):
        discrete_distrib = IIDStdGaussian()
        discrete_distrib.mimics = 'Poisson' # Transform from poisson to Gaussion not implemented yet
        true_distrib = Gaussian(2)
        self.assertRaises(TransformError, true_distrib.transform_generator, discrete_distrib)

class Test_Uniform(unittest.TestCase):
    
    def test_transforms(self):
        a,b = 1/4,1/2
        true_distrib = Uniform(3,lower_bound=a,upper_bound=b)
        true_distrib.transform_generator(IIDStdUniform()) # IIDStdUniform -> Uniform(1/4,1/2)
        vals = true_distrib[0].gen_distribution(5,4)
        with self.subTest(): self.assertTrue((vals<=b).all() and (vals>=a).all())
        true_distrib = Uniform(3,lower_bound=a,upper_bound=b)
        true_distrib.transform_generator(IIDStdGaussian()) # IIDStdGaussian -> Uniform(1/4,1/2)
        vals = true_distrib[0].gen_distribution(5,4)
        with self.subTest(): self.assertTrue((vals<=b).all() and (vals>=a).all())

class Test_Gaussian(unittest.TestCase):
    
    def test_transforms(self):
        pass

class Test_BrownianMontion(unittest.TestCase):
    
    def test_transforms(self):
        pass

if __name__ == "__main__":
    unittest.main()
