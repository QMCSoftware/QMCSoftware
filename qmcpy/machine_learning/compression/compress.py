#from ..util import MethodImplementationError, _univ_repr, ParameterError
from typing import Any
import numpy as np
#from .myhosobol import MyHOSobol
#from .approxmeanMXY import approxmeanMXY
import logging


class Compress:
		'''
		Abstract class for quasi-monto data compression.
		Computes the weights W_X,Y and W_X.
		nu ... \nu in the paper
		m ... \ell in the paper
		s ... dimension of data (this is in the underlying implementation and isn't in this abstract class)
		N ... number of datapoints
		Nqmc ... number of qmc points
		outs ... output dimension of the neural network (weights W_X,Y are vector valued if outs>1)
		px ... pointer to datapoints array
		pz ... pointer to qmc points array
		py ... pointer to y values array

		Output is a pointer to a vector which contains the weights W_X (Nqmc entries), and then the dimensions of W_X,Y (Nqmc x outs entries)
		in the same order as the qmc points.


		Output is a pointer to a vector which contains the weights W_X (Nqmc entries),
		and then the dimensions of W_X,Y (Nqmc x outs entries)  in the same order as the qmc points.
		Refrences:
		[1] J. Dick, M. Feischl, A quasi-Monte Carlo data compression algorithm for machine learning, Journal of Complexity, https://doi.org/10.1016/j.jco.2021.101587
		'''

		def __init__(self,
					 nu: int,
					 m: int,
					 input_file: str,
					 label_file: str,
					 d: int,):
					 # weights:str,
					 #z: str
					 #):
					"""
					Args:
						dimension (int or ndarray): dimension of the generator.
						If an int is passed in, use sequence dimensions [0,...,dimensions-1].
						If a ndarray is passed in, use these dimension indices in the sequence.
						Note that this is not relevant for IID generators.
					seed (int or numpy.random.SeedSequence): seed to create random number generator
					"""
					prefix = 'A concrete implementation of DiscreteDistribution must have '
					self.nu = nu
					self.m  = m
					self.input_file = input_file
					self.lable_file = label_file
					self.d = d
					
					if not hasattr(self, 'nu'):
						raise ParameterError(prefix + 'self.nu (measure mimiced by the distribution)')
					if not hasattr(self, 'm'):
						raise ParameterError(prefix + 'self.m (measure mimiced by the distribution)')
					if not hasattr(self, 'input_file'):
						raise ParameterError(prefix + 'self.input_file (Input data)')
					if not hasattr(self, 'label_file'):
						raise ParameterError(prefix + 'self.label_file (Labels for Input Data)')
					if not hasattr(self, 'd'):
						raise ParameterError(prefix + 'self.d (measure mimiced by the distribution)')



		def load_data(self):
			""" ABSTRACT METHOD to get the input (dataset) """
			raise MethodImplementationError(self, 'load_data')

		def load_labels(self):
			""" ABSTRACT METHOD to get the sobol sequence to compute weights. """
			raise MethodImplementationError(self, 'load_labels')

		def get_sobol(self ):
			""" ABSTRACT METHOD to get the sobol sequence to compute weights. """
			raise MethodImplementationError(self, 'get_sobol')

		def compute_weights(self):
			""" ABSTRACT METHOD to compute weights. """
			raise MethodImplementationError(self, 'compute_weights')

		def __repr__(self):
			return _univ_repr(self, "_compressor", self.parameters)