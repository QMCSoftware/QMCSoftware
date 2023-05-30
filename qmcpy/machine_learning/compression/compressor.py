import logging
import numpy as np
from myhosobol import MyHOSobol
from approxmeanMXY import approxmeanMXY
from compress import Compress

class digital_net_compression(Compress):

	"""
	>>> import compressor
	>>> test = compressor.digital_net_compression(nu = 1, m=6, input_file="./test_data/reg_x.csv",label_file="./test_data/reg_y.csv" ,d=1)
	>>> test.compute_weights()
	Computation complete
	"""

	def __init__(self,
		     nu: int,
		     m: int,
		     input_file: str,
		     label_file: str,
                     d: int,):
				 #weights: str,
				 #z: str):
		self.nu = nu
		self.m  = m
		self.input_file = input_file
		self.lable_file = label_file
		self.d = d
		#weights = weights
		#z = z

	def load_inputs(self):
		return np.loadtxt(self.input_file, delimiter=",")
		# logging.debug(self.dataset)
		# return self.dataset

	def load_labels(self):
		return np.loadtxt(self.label_file, delimiter=",")
		# return self.labels

	def get_sobol(self):
		"""Eventually we will replace this with the internal sobol library"""
		# z = MyHOSobol(self.m, self.s, self.d)
		# return self.sobol
		pass

	def compute_weights(self):
		#logging.debug(self.load_inputs())
		#logging.debug(self.load_labels())

		weights, z = approxmeanMXY(
			self.nu, self.m, self.load_inputs(), self.load_labels(), self.d
		)
		np.savetxt(
			"weights_computed.npy", self.weights
		)  # TODO: Relative path with date and time, don't save into current pip install, make it not overwrite existing files.
		np.savetxt(
			"linear_regression_compresed.npy", self.z
		)  # TODO: Relative path with date and time, don't save into current pip install, make it not overwrite existing files.
		print("Computation complete")


if __name__ == "__main__":
	import logging

	logging.basicConfig()
	logging.getLogger().setLevel(logging.DEBUG)
	import doctest

	doctest.testmod()
