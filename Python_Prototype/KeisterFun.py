from fun import fun
from numpy import square, cos, exp, sqrt, multiply, sum
import numpy as np


class KeisterFun(fun):

    # Â§\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$Â§
    def f(self, x, coordIndex):
        """
        >>> kf = KeisterFun()
        >>> kf.f(np.array([[1, 2], [3, 4]]), [1, 2])
        array([ -4.15915193e-03,   3.93948451e-12])
        """
        # if the nominalValue = 0, this is efficient
        normx2 = sum(square(x), axis=1)
        if (len(coordIndex) != self.dimension) and (self.nominalValue != 0):
            normx2 = normx2 + square(self.nominalValue) * (self.dimension - len(coordIndex))
        y = multiply(exp(-normx2), cos(sqrt(normx2)))
        return y


if __name__ == "__main__":
    import doctest

    doctest.testmod()
