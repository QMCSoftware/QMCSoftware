from . import Integrand

class Linear(Integrand):

    def __init__(self, nominal_value=None):
        super().__init__(nominal_value=nominal_value)

    def g(self, x, coord_index):
        y = x.sum(1)
        return y