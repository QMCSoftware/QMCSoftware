from fun import fun
from numpy import square, cos, exp, sqrt, multiply


class KeisterFun(fun):


    # Â§\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$Â§
    def f(self, x, coordIndex):
        # if the nominalValue = 0, this is efficient
        normx2 = sum(square(x), 2)
        if (len(coordIndex) != self.dimension) and (self.nominalValue != 0):
            normx2 = normx2 + square(self.nominalValue) * (self.dimension - len(coordIndex))
        y = multiply(exp(-normx2), cos(sqrt(normx2)))
        return y

