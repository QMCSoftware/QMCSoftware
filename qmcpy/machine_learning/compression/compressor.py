import logging
from dataclasses import dataclass
from myhosobol import MyHOSobol
from typing import Any
from approxmeanMXY import approxmeanMXY
import numpy as np

@dataclass
class digital_net_compression:

	'''
	Args:
	Computes the weights W_X,Y and W_X.
	nu ... \nu in the paper
	m ... \ell in the paper
	s ... dimension of data
	N ... number of datapoints
	Nqmc ... number of qmc points
	outs ... output dimension of the neural network (weights W_X,Y are vector valued if outs>1)
	px ... pointer to datapoints array
	pz ... pointer to qmc points array
	py ... pointer to y values array

	Output is a pointer to a vector which contains the weights W_X (Nqmc entries), and then the dimensions of W_X,Y (Nqmc x outs entries) in the same order as the qmc points.


	Output is a pointer to a vector which contains the weights W_X (Nqmc entries),
	and then the dimensions of W_X,Y (Nqmc x outs entries)  in the same order as the qmc points.
	'''
	nu: int
	m: int
	input_file: str
	label_file: str
	d: int
	# def __init__(self, nu , m, s, N, Ndata, Nqmc, output_dimentsion, dataset):

	#		self.m = m #ell in the paper
	#		self.nu = nu
	#		self.Ndata = Ndata
	#		self.s = s
	#		self.Nqmc = Nqmc	#		self.dataset = dataset
	#		self.labels  = labels
	#		self.sobol   = sobol
	#		self.outputfile = outputfile
	#		# load c functions

	def load_inputs(self):
		return np.loadtxt(self.input_file, delimiter=',')
		#logging.debug(self.dataset)
		#return self.dataset

	def load_labels(self):
		return np.loadtxt(self.label_file, delimiter=',')
		#return self.labels

	def get_sobol(self ):
		'''Eventually we will replace this with the internal sobol library'''
		#z = MyHOSobol(self.m, self.s, self.d)
		#return self.sobol
		pass

	def compute_weights(self):
		#x,  z , y = self.get_dataset()
		#return computeMXY(self.nu, self.m, self.base, input, self.get_sobol(), self.labels)
		weights, z = approxmeanMXY(self.nu,self.m,self.load_inputs(), self.load_labels(), self.d) 
		np.savetxt('weights_computed.npy', weights)
		np.savetxt('linear_regression_compresed.npy', z)
		print('Computation complete')
