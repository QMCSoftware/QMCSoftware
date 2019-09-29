''' Originally developed in MATLAB by Fred Hickernell. Translated to python by
Sou-Cheng T. Choi and Aleksei Sorokin '''
from numpy import cos, pi

from . import Fun


class KeisterFun(Fun):
    '''
    Specify and generate values $f(\vx) = \pi^{d/2} \cos(\lVert \vx \rVert)$ for $\vx \in \reals^d$Â§
    The standard example integrates the Keister function with respect to an IID Gaussian distribution with variance 1/2
    B. D. Keister, Multidimensional Quadrature Algorithms, Â§\mcommentfont \emph{Computers in Physics}, \textbf{10}, pp.\ 119-122, 1996.
    '''
    
    def __init__(self,nominalValue=None):
        super().__init__(nominalValue=nominalValue)
        
    def g(self,x,coordIndex):
        # if the nominalValue = 0, this is efficient
        normx2 = (x**2).sum(1)
        nCoordIndex = len(coordIndex)
        if nCoordIndex != self.dimension and self.nominalValue != 0:
            normx2 = normx2 + self.nominalValue**2 * (self.dimension - nCoordIndex)
        y = pi**(nCoordIndex/2)*cos(normx2**.5)
        return y