from numpy import square, cos, exp, sqrt, multiply, sum, array
from fun import fun as fun


class KeisterFun(fun):
    def __init__(self):
        super().__init__()

    # Specify and generate values $f(\vx)$ for $\vx \in \cx$
    def f(self, x, coordIndex):
        # if the nominalValue = 0, this is efficient
        normx2 = sum(square(x), axis=1)
        coordIndex = array(coordIndex)
        if (coordIndex.size != self.dimension) and (self.nominalValue != 0):
            normx2 = normx2 + square(self.nominalValue) * (self.dimension - coordIndex.size)
        y = multiply(exp(-normx2), cos(sqrt(normx2)))
        return y
    
if __name__ == "__main__":
    # Doctests
    import doctest
    x = doctest.testfile("dt_Keister.py")
    print("\n"+str(x))