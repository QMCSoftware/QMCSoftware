from ._integrand import Integrand


class CustomFun(Integrand):
    """ Specify and generate values of a user-defined function"""

    def __init__(self, measure, custom_fun):
        """
        Args:
            measure (TrueMeasure): a TrueMeasure instance
            custom_fun (function): a function evaluating samples (nxd) -> (nx1). See g method.
        """
        self.measure = measure
        self.custom_fun = custom_fun
        super().__init__()

    def g(self, x):
        """ See abstract method. """
        return self.custom_fun(x)
