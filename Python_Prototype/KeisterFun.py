from numpy import square, cos, exp, sqrt, multiply, sum, array, pi
from fun import fun

class KeisterFun(fun):
    '''
    Â§\mcommentfont Specify and generate values $f(\vx) = \pi^{d/2} \cos(\lVert \vx \rVert)$ for $\vx \in \reals^d$Â§
    The standard example integrates the Keister function with respect to an IID Gaussian distribution with variance 1/2
    B. D. Keister, Multidimensional Quadrature Algorithms, Â§\mcommentfont \emph{Computers in Physics}, \textbf{10}, pp.\ 119-122, 1996.Â§
    '''
    def __init__(self):
        super().__init__()
        
    # Specify and generate values $f(\vx)$ for $\vx \in \cx$
    def g(self,x,coordIndex):
        # if the nominalValue = 0, this is efficient
        normx2 = (x**2).sum(1)
        nCoordIndex = len(coordIndex)
        if nCoordIndex != self.dimension and self.nominalValue != 0:
            normx2 = normx2 + self.nominalValue**2 * (self.dimension - nCoordIndex)
        y = pi**(nCoordIndex/2)*cos(normx2**.5)
        return y

if __name__ == "__main__":
    # Doctests
    import doctest
    x = doctest.testfile("Tests/dt_KeisterFun.py")
    print("\n"+str(x))