""" Definition of Lebesgue, a concrete implementation of TrueMeasure """

from ._true_measure import TrueMeasure

from numpy import array


class Lebesgue(TrueMeasure):
    """ Lebesgue Uniform Measure """

    def __init__(self, dimension, lower_bound=0., upper_bound=1):
        """
        Args:
            dimension (ndarray): dimension's' of the integrand's'
        """
        transforms = {
            "StdUniform": [
                lambda self, samples: samples * (self.b - self.a) + self.a,
                # stretch samples
                lambda self, g: g * (self.b - self.a).prod()]}  # multiply dimensional difference
        super().__init__(dimension, [transforms],
                         a=lower_bound,
                         b=upper_bound)

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        return super().__repr__(['a', 'b'])
