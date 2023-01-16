
import numpy as np
def MyHOSobol(m, s, d=1):
	# Higher order Sobol sequence
	# Create a higher order Sobol sequence.
	# 2^m number of points
	# s dimension of final point set
	# d interlacing factor
	# X Output Sobol sequence
	z = np.loadtxt('sobol.dat')
	z = z[:2**m, :s*d]
	if d > 1:
		print("Please contact the QMCPy team for this use case")
		"""
		N = pow(2, m) # Number of points;
		u = 52
		depth = np.floor(u/d)
		
		# Create binary representation of digits;
		
		W = z * pow(2, np.int64(depth))
		Z = np.floor(np.transpose(W))
		Y = np.zeros((s, N))
		for j in range(s):
			for i in range(depth):
				for k in range(d):
					Y[j,:] = np.bitset(Y[j,:], (depth*d+1) - k - (i-1)*d, np.bitget(Z[(j-1)*d+k,:], (depth+1) - i))
		Y = Y * pow(2, -depth*d)

		X = np.transpose(Y) # X is matrix of higher order Sobol points,
		# where the number of columns equals the dimension
		# and the number of rows equals the number of points;
		"""
	else:
		X = z
		
	return X