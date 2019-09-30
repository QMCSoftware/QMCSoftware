import unittest

from numpy import arange
from algorithms.distribution.Measures import BrownianMotion

class Test_Measure(unittest.TestCase):

    def test_measure_construction(self):
        measure = BrownianMotion(timeVector=[arange(1/4,5/4,1/4),arange(1/16,17/16,1/16)])
        with self.subTest(): self.assertEqual(len(measure),2)
        with self.subTest(): self.assertEqual(measure[1].dimension,16)
        with self.subTest(): self.assertEqual(type(measure[1]).__name__,'BrownianMotion')
        with self.subTest(): self.assertTrue(hasattr(measure[1],'timeVector'))
        with self.subTest(): self.assertEqual(measure[1].timeVector[0],1/16)

if __name__ == "__main__":
    unittest.main()