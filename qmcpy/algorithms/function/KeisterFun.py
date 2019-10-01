from numpy import cos, pi

from . import Fun


class KeisterFun(Fun):
    '''
    Specify and generate values :math:`f(\mathbf{x}) = \pi^{d/2} \cos(\| \mathbf{x} \|)` for :math:`\mathbf{x} \in \mathbb{R}^d`

    The standard example integrates the Keister function with respect to an IID Gaussian distribution with variance 1/2

    B. D. Keister, Multidimensional Quadrature Algorithms,  \emph{Computers in Physics}, \textbf{10}, pp.\ 119-122, 1996.
    '''
    
    def __init__(self, nominal_value=None):
        super().__init__(nominal_value=nominal_value)

    def g(self, x, coordIndex):
        # if the nominalValue = 0, this is efficient
        normx2 = (x**2).sum(1)
        nCoordIndex = len(coordIndex)
        if nCoordIndex != self.dimension and self.nominalValue != 0:
            normx2 = normx2 + self.nominalValue**2 * (self.dimension - nCoordIndex)
        y = pi**(nCoordIndex/2)*cos(normx2**.5)
        return y