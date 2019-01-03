from numpy import square, cos, exp, sqrt, multiply, sum, array, pi
from fun import fun


class KeisterFun(fun):
    def __init__(self):
        super().__init__()
        self.distrib['name'] = 'IIDZGaussian'
        self.distrib['variance'] = 0.5
        
    # Specify and generate values $f(\vx)$ for $\vx \in \cx$
    def g(self, x, coordIndex):
        # if the nominalValue = 0, this is efficient
        normx2 = sum(square(x), axis=1)
        coordIndex = array(coordIndex)
        nCoordIndex = len(coordIndex)
        if nCoordIndex != self.dimension and self.nominalValue != 0:
            normx2 = normx2 + square(self.nominalValue) * (self.dimension - nCoordIndex)
        y = multiply(pi**(nCoordIndex/2), cos(sqrt(normx2)))
        return y

if __name__ == "__main__":
    # Doctests
    import doctest
    x = doctest.testfile("Tests/dt_KeisterFun.py")
    print("\n"+str(x))