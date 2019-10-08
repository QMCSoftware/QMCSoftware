""" Definitions for concrete Measure Implementations """

from numpy import arange

from . import Measure


class StdUniform(Measure):
    """ Standard Uniform Measure """

    def __init__(self, dimension=None):
        """
        Args:
            dimension (array of ints): dimensions of integrands
        """
        super().__init__(dimension)


class StdGaussian(Measure):
    """ Standard Gaussian Measure """

    def __init__(self, dimension=None):
        """
        Args:
            dimension (array of ints): dimensions of integrands
        """
        super().__init__(dimension)


class IIDZeroMeanGaussian(Measure):
    """ IID Zero Mean Gausian Measure """

    def __init__(self, dimension=None, variance=None):
        """
        Args:
            dimension (array of ints): dimensions of integrands
            variance (array of floats): variance of each gaussian distribution
        """
        super().__init__(dimension, variance=variance)


class BrownianMotion(Measure):
    """ Brownian Motion Measure """

    def __init__(self, time_vector=None):
        """
        Args:
            time_vector (array of floats): Monitoring times
        """
        if time_vector:
            dimension = [len(tV) for tV in time_vector]
                   # dimensions of each integrand
        else:
            dimension = None
        super().__init__(dimension, time_vector=time_vector)


class Lattice(Measure):
    """ Lattice (Base 2) Measure """

    def __init__(self, dimension=None):
        """
        Args:
            dimension (array of ints): dimensions of integrands
        """
        super().__init__(dimension, mimics="StdUniform")


class Sobol(Measure):
    """ Sobol (Base 2) Measure """

    def __init__(self, dimension=None):
        """
        Args:
            dimension (array of ints): dimensions of integrands
        """
        super().__init__(dimension, mimics="StdUniform")
