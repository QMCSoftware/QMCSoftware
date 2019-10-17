""" Utility methods. Not meant for public use """


def univ_repr(qmc_object, attributes=None):
    """Clean way to represent qmc_object data.

    Note: ::

        print(qmc_object)

    is equivalent to ::

        print(qmc_object.__repr__())

    Args:
        qmc_object: an qmc_object instance
        attributes: list of qmc_object attribute names whose values are to be
        gathered

    Returns:
        str

    """
    key_val = "%s qmc_object with properties:\n" % (type(qmc_object).__name__)
    for key, val in qmc_object.__dict__.items():
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
    for i, sub_obj in enumerate(qmc_object):
        key_val += "%8s%s[%d] with properties:\n" % ("", attributes, i)
        for key, val in sub_obj.__dict__.items():
            if str(key) != attributes:
                key_val += "%12s%s: %s\n" % (
                    "",
                    str(key),
                    str(val).replace("\n", "\n%20s" % ("")),
                )
    return key_val[:-1]


def summarize(stopping_criterion=None, measure=None, integrand=None, distribution=None, data=None):
    """Print a summary of inputs and outputs for the qmc problem after execution.

    Args:
        stopping_criterion (StoppingCriterion): a Stopping Criterion object
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
    if not stopping_criterion is None:
        stopping_criterion.summarize()
    if not data is None:
        data.summarize()



# API
from .exceptions_warnings import *
