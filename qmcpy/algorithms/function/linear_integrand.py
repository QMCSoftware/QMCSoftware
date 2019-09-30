import numpy as np

from algorithms.function.integrand_base import IntegrandBase


class LinearFun(IntegrandBase):
    def g(self, x, coords_in_sequence):
        return np.sum(x, axis=1)
