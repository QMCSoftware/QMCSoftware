import unittest

from numpy import arange
from algorithms.distribution import Measure

class Test_Measure(unittest.TestCase):

    def test_std_uniform(self):
        measure = Measure().std_uniform(dimension=[2,3,4])
        with self.subTest(): self.assertEqual(len(measure),3)
        with self.subTest(): self.assertEqual(measure[1].dimension,3)
        with self.subTest(): self.assertEqual(measure[1].measureName,'std_uniform')
    
    def test_std_gaussian(self):
        measure = Measure().std_gaussian(dimension=[2,3,4])
        with self.subTest(): self.assertEqual(len(measure),3)
        with self.subTest(): self.assertEqual(measure[1].dimension,3)
        with self.subTest(): self.assertEqual(measure[1].measureName,'std_gaussian')
    
    def test_iid_zmean_gaussian(self):
        measure = Measure().iid_zmean_gaussian(dimension=[2,3,4],variance=[1/4,1/2,3/4])
        with self.subTest(): self.assertEqual(len(measure),3)
        with self.subTest(): self.assertEqual(measure[1].dimension,3)
        with self.subTest(): self.assertEqual(measure[1].measureName,'iid_zmean_gaussian')
        with self.subTest(): self.assertEqual(measure[1].measureData['variance'],1/2)

    def test_brownian_motion(self):
        tV = [arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)]
        measure = Measure().brownian_motion(timeVector=tV)
        with self.subTest(): self.assertEqual(len(measure),3)
        with self.subTest(): self.assertEqual(measure[1].dimension,16)
        with self.subTest(): self.assertEqual(measure[1].measureData['timeVector'][0],1/16)
        with self.subTest(): self.assertEqual(measure[1].measureName,'brownian_motion')

    def test_lattice(self):
        measure = Measure().lattice(dimension=[2,3,4])
        with self.subTest(): self.assertEqual(len(measure),3)
        with self.subTest(): self.assertEqual(measure[1].dimension,3)
        with self.subTest(): self.assertEqual(measure.measureName,'lattice')
        with self.subTest(): self.assertEqual(measure[1].measureName,'std_uniform')
        with self.subTest(): self.assertEqual(measure[1].measureData['lds_type'],'lattice')
    
    def test_sobol(self):
        measure = Measure().sobol(dimension=[2,3,4])
        with self.subTest(): self.assertEqual(len(measure),3)
        with self.subTest(): self.assertEqual(measure[1].dimension,3)
        with self.subTest(): self.assertEqual(measure.measureName,'sobol')
        with self.subTest(): self.assertEqual(measure[1].measureName,'std_uniform')
        with self.subTest(): self.assertEqual(measure[1].measureData['lds_type'],'sobol')

if __name__ == "__main__":
    unittest.main()