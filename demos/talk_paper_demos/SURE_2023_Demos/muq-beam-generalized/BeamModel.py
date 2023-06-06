import muq.Modeling as mm
import numpy as np

class EulerBernoulli(mm.PyModPiece):
    """ Computes the displacement of an Euler-Bernoulli beam using a
        second order finite difference scheme.  The beam is assumed
        to be in a cantilever configuration, where the left boundary 
        is fixed (u=0, du/dx=0) and the right boundary is  free (d2u/dx2=0, d3u/dx3=0).
        
        This class can be used in two ways:
        1) Setting the stiffness in the constructor, in which case the ModPiece
           will have a single input: the load.  In this case, the stiffness 
           matrix is precomputed and can be accessed 
        2) Setting the stiffness during the call to Evaluate, in which case
           there will be two inputs: [load, stiffness]
           
    """
    
    def __init__(self, numNodes, length, radius, constMod=np.array([])):
        if constMod.shape[0]==0:
            mm.PyModPiece.__init__(self, [numNodes, numNodes], # Two inputs (load, stiffness)
                                          [numNodes]) # One output (the displacement)
        else:
            mm.PyModPiece.__init__(self, [numNodes], # One input (the load)
                                          [numNodes])# One output (the displacement)
        self.numNodes = numNodes

        self.dx = length/(numNodes-1)

        # Moment of inertia assuming cylindrical beam
        self.I = np.pi/4.0*radius**4

        if(constMod.shape[0]>0):
            self.K = self.BuildK(constMod)

    def BuildK(self, modulus):
        # Number of nodes in the finite difference stencil
        n = self.numNodes

        # Create stiffness matrix
        K = np.zeros((n, n))

        # Build stiffness matrix (center)
        for i in range(2, n-2):
            K[i,i+2] = modulus[i]
            K[i,i+1] = modulus[i+1] - 6.0*modulus[i] + modulus[i-1]
            K[i,i]   = -2.0*modulus[i+1] + 10.0*modulus[i] - 2.0*modulus[i-1]
            K[i,i-1] = modulus[i+1] - 6.0*modulus[i] + modulus[i-1]
            K[i,i-2] = modulus[i]

        # Set row i == 1
        K[1,3] = modulus[1]
        K[1,2] = modulus[2] - 6.0*modulus[1] + modulus[0]
        K[1,1] = -2.0*modulus[2] + 11.0*modulus[1] - 2.0*modulus[0]

        # Set row i == n - 2
        K[n-2,n-1] = modulus[n-1] - 4.0*modulus[n-2] + modulus[n-3]
        K[n-2,n-2] = -2.0*modulus[n-1] + 9.0*modulus[n-2] - 2.0*modulus[n-3]
        K[n-2,n-3] = modulus[n-1] - 6.0*modulus[n-2] + modulus[n-3]
        K[n-2,n-4] = modulus[n-2]

        # Set row i == n - 1 (last row)
        K[n-1,n-1] = 2.0*modulus[n-1]
        K[n-1,n-2] = -4.0*modulus[n-1]
        K[n-1,n-3] = 2.0*modulus[n-1]

        # Apply dirichlet BC (w=0 at x=0)
        K[0,:] = 0.0; K[:,0] = 0.0
        K[0,0] = 1

        return K/self.dx**4

    def EvaluateImpl(self, inputs):
        # Distributed load over the beam (np.array)
        load = inputs[0]/self.I

        # Apply dirichlet BC on load vector
        load[0] = 0

        if self.inputSizes.shape[0]>1:
            modulus = inputs[1]
            self.K = self.BuildK(modulus)

        # Solve system to get displacement
        displacement = np.linalg.solve(self.K, load)
        self.outputs = [displacement]


if __name__=='__main__':
    import h5py 

    f = h5py.File('ProblemDefinition.h5','r')

    x = np.array( f['/ForwardModel/NodeLocations'] )
    
    length = f['/ForwardModel'].attrs['BeamLength']
    radius = f['/ForwardModel'].attrs['BeamRadius']

    loads = np.array( f['/ForwardModel/Loads'])

    forwardMod = EulerBernoulli(x.shape[1], length, radius)

    loads = mm.ConstantVector(loads)

    graph = mm.WorkGraph()
    graph.AddNode(forwardMod, 'Forward Model')
    graph.AddNode(loads,'Loads')
    graph.AddEdge('Loads',0,'Forward Model', 0)

    mod = graph.CreateModPiece('Forward Model')

    mm.serveModPiece(mod, "forward", "0.0.0.0", 4242)
