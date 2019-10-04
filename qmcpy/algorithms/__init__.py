"""
Utilities for qmcpy
"""


# Exceptions
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


# Warnings
class MaxSamplesWarning(Warning):
    """
    Class for issuing warning about using maximum number of data samples
    """


def univ_repr(object, attributes=None):
    """
    Clean way to represent object data.

    Note: ::

        print(object)

    is equivalent to ::

        print(object.__repr__())

    Args:
        object: an object instance
        attributes: list of object attribute names whose values are to be gathered

    Returns:
        str

    """
    key_val = '%s object with properties:\n' % (type(object).__name__)
    for key, val in object.__dict__.items():
        if str(key) != attributes:
            key_val += '%4s%s: %s\n' % \
                 ('', str(key), str(val).replace('\n', '\n%15s' % ('')))
    if not attributes:
        return key_val[:-1]

    # print list of subObject with properties
    key_val += '    %s:\n' % (attributes)
    for i, sub_obj in enumerate(object):
        key_val += '%8s%s[%d] with properties:\n' % ('', attributes, i)
        for key, val in sub_obj.__dict__.items():
            if str(key) != attributes:
                key_val += '%12s%s: %s\n' % \
                     ('', str(key), str(val).replace('\n', '\n%20s' % ('')))
    return key_val[:-1]
