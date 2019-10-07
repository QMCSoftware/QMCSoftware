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


class TransformError(Exception):
    """
    Class for raising error about transforming function to accomodate \
    distributuion
    """


class MaxSamplesWarning(Warning):
    """
    Class for issuing warning about using maximum number of data samples
    """
