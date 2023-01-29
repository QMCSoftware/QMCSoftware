import unittest
import numpy as np
import ..linreg

class TestLinreg(unittest.TestCase):

	def testLinRegRun(self):
		#pass
		reg_x = np.loadtxt("reg_x.csv", delimiter=",", dtype=str)
		reg_y = np.loadtxt("reg_y.csv", delimiter=",", dtype=str)
		reg_z = np.loadtxt("reg_y.csv", delimiter=",", dtype=str)
		reg_weights = np.loadtxt("reg_weights.csv",    delimiter=",", dtype=str)

		#breakpoint()




