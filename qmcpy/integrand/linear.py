from ._integrand import Integrand


class Linear(Integrand):
    """ $f(\\boldsymbol{x}) = \\sum_{i=1}^d x_i$ """

    def __init__(self, measure):
        """
        Args:
            measure (TrueMeasure): a TrueMeasure instance
        """
        self.measure = measure
        super().__init__()

    def g(self, x):
        """ See abstract method. """
        y = x.sum(1)  # Linear sum
        return y
