from . import Integrand

class Linear(Integrand):
    """ Specify and generate values :math:`f(\mathbf{x}) = \sum_{i=1}^d x_i` for :math:`\mathbf{x} = (x_1,\ldots,x_d) \in \mathbb{R}^d`."""
    def __init__(self, nominal_value=None):
        super().__init__(nominal_value=nominal_value)

    def g(self, x, coord_index):
        y = x.sum(1)
        return y