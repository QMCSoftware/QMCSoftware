""" Utility methods. Not meant for public use """


def univ_repr(object, attributes=None):
    """Clean way to represent object data.

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
    key_val = "%s object with properties:\n" % (type(object).__name__)
    for key, val in object.__dict__.items():
        if str(key) != attributes:
            key_val += "%4s%s: %s\n" % (
                "",
                str(key),
                str(val).replace("\n", "\n%15s" % ("")),
            )
    if not attributes:
        return key_val[:-1]

    # print list of subObject with properties
    key_val += "    %s:\n" % (attributes)
    for i, sub_obj in enumerate(object):
        key_val += "%8s%s[%d] with properties:\n" % ("", attributes, i)
        for key, val in sub_obj.__dict__.items():
            if str(key) != attributes:
                key_val += "%12s%s: %s\n" % (
                    "",
                    str(key),
                    str(val).replace("\n", "\n%20s" % ("")),
                )
    return key_val[:-1]


# API
from .exceptions_warnings import *


def summarize(stop=None, measure=None, integrand=None, distribution=None, data=None):
    """Print a summary of inputs and outputs for the qmc problem after execution.

    Args:
        stop (StoppingCriterion): a Stopping Criterion object
        measure (Measure): a Measure object
        integrand (Integrand): an Integrand object
        data (data): a AccumData object
    """
    if not integrand is None:
        integrand.summarize()
    if not measure is None:
        measure.summarize()
    if not distribution is None:
        distribution.summarize()
    if not stop is None:
        stop.summarize()
    if not data is None:
        data.summarize()
    return