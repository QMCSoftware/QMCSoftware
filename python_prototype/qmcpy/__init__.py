"""
Utilities for python_prototype
"""
name = "qmcpy"
__version__ = 0.1

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


def print_summary(stopObj, measureObj, funObj, distribObj, dataObj):
    h1 = '%s (%s)\n'
    item_i = '%25s: %d\n'
    item_f = '%25s: %-15.4f\n'
    item_s = '%25s: %-15s\n'
    s = 'Solution: %-15.4f\n%s\n'%(dataObj.solution,'~'*50)

    s += h1%(type(funObj).__name__,'Function Object')

    s += h1%(type(measureObj).__name__,'Measure Type')
    s += item_s%('dimension',str(measureObj.dimension))

    s += h1%(type(distribObj).__name__,'Distribution Object')
    s += item_s%('true_distribution.measureName',type(distribObj.true_distribution).__name__)

    s += h1%(type(stopObj).__name__,'StoppingCriterion Object')
    s += item_f%('abs_tol',stopObj.abs_tol)
    s += item_f%('rel_tol',stopObj.rel_tol)
    s += item_i%('n_max',stopObj.n_max)
    s += item_f%('inflate',stopObj.inflate)
    s += item_f%('alpha',stopObj.alpha)

    s += h1%(type(dataObj).__name__,'Data Object')
    s += item_s%('n_samples_total',str(dataObj.n_samples_total))
    s += item_f%('t_total',dataObj.t_total)
    s += item_s%('confid_int',str(dataObj.confid_int))

    print(s)
    return