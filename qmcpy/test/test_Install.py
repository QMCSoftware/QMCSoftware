import unittest

from algorithms.distribution import measure
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.function.KeisterFun import KeisterFun
from algorithms.integrate import integrate
from algorithms.stop.CLTStopping import CLTStopping

class IntegrationExampleTest(unittest.TestCase):
    '''
    def test_qmcpy_version(self):
        import qmcpy
        self.assertEqual(qmcpy.__version__, 0.1)
    '''


if __name__ == "__main__":
    unittest.main()
