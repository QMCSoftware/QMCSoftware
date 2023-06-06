import numpy as np
import random
import h5py

# MUQ Includes
import muq.Modeling as mm # Needed for Gaussian distribution
import muq.Approximation as ma # Needed for Gaussian processes

# Include the beam model
from BeamModel import EulerBernoulli

# Discretization
numPts = 31
dim = 1

length = 1.0
x = np.linspace(0,length,numPts)[None,:]

# Geometry of beam (assumes beam is cylinder with constant cross sectional area)
radius = 0.1

# Create the beam model.  Note that the beam class has a member "K" holding the stiffness
# matrix.  Thus, "beam.K" will give you access to the matrix K referenced above
beam = EulerBernoulli(numPts, length, radius)


numObs = 20
B = np.zeros((numObs,numPts))

obsInds = random.sample(range(numPts), numObs)
for i in range(numObs):
    B[i,obsInds[i]] = 1.0

# Define the prior over the loads and generate a random realization
priorVar = 10*10
priorLength = 0.5
priorNu = 3.0/2.0 # must take the form N+1/2 for zero or odd N (i.e., {0,1,3,5,...})

kern1 = ma.MaternKernel(1, 1.0, priorLength, priorNu)
kern2 = ma.ConstantKernel(1, 10*10)
kern = kern1 + kern2

mu = ma.ZeroMean(1,1)
priorGP = ma.GaussianProcess(mu,kern)
q = priorGP.Sample(x)[0,:]

# Define the prior of the stiffness and generate a random realization
kern = ma.MaternKernel(1, 4, 0.2, 3.0/2.0)
priorGP = ma.GaussianProcess(mu,kern)
E = np.exp(priorGP.Sample(x)[0,:]+10)

u = beam.Evaluate([q,E])[0]
print('u.shape = ', u.shape)

# Open an HDF5 file for saving
fObs = h5py.File('ProblemDefinition.h5','w')

fObs['/ForwardModel/Loads'] = q
fObs['/ForwardModel/NodeLocations'] = x
fObs['/ForwardModel/Modulus'] = E
fObs['/ForwardModel/TrueDisplacement'] = u
fObs['/ForwardModel'].attrs['BeamLength'] = length
fObs['/ForwardModel'].attrs['BeamRadius'] = radius

fObs['/Observations/ObservationMatrix'] = B 
fObs['/Observations/ObservationData'] = np.dot(B,u)

print(q)


