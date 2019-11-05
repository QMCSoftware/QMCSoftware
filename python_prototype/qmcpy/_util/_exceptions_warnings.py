""" Exceptions and Warnings thrown by qmcpy """


class MeasureCompatibilityError(Exception):
    """
    Class for raising error of incompatible measures
    """


class DimensionError(Exception):
    """
    Class for raising error about dimension
    """


class DistributionCompatibilityError(Exception):
    """
    Class for raising error about incompatible distribution
    """
class NotYetImplemented(Exception):
    """
    Class for raising error when a component has been implemented yet
    """

class TransformError(Exception):
    """
    Class for raising error about transforming function to accommodate \
    distribution
    """

class ParameterError(Exception):
    """
    Class for raising error about input parameters
    """
class ParameterWarning(Warning):
    """
    Class for issuing warnings about unacceptable parameters
    """

class MaxSamplesWarning(Warning):
    """
    Class for issuing warning about using maximum number of data samples
    """
