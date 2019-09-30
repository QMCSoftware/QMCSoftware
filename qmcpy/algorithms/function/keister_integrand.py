''' Originally developed in MATLAB by Fred Hickernell. Translated to python by
Sou-Cheng T. Choi and Aleksei Sorokin '''
import numpy as np

from algorithms.function.integrand_base import IntegrandBase


class KeisterFun(IntegrandBase):
    '''
    Specify and generate values $f(\vx) = \pi^{d/2} \cos(\lVert \vx \rVert)$ for $\vx \in \reals^d$
    The standard example integrates the Keister function with respect to an IID Gaussian distribution with variance 1/2
    B. D. Keister, Multidimensional Quadrature Algorithms,  \emph{Computers in Physics}, \textbf{10}, pp.\ 119-122, 1996.
    '''

    def g(self, x, coords_in_sequence):
        # if the nominalValue = 0, this is efficient
        sum_of_squares = np.sum(x ** 2, axis=1)
        num_coords_active = len(coords_in_sequence)
        if num_coords_active != self.dimension and self.nominal_value != 0:  # This if-block seems unnecessary
            sum_of_squares += self.nominal_value ** 2 * (self.dimension - num_coords_active)
        return np.pi ** (num_coords_active / 2) * np.cos(sum_of_squares ** .5)
