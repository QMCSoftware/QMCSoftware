""" Utility functions that abstract QMCPy objects. """

from numpy import array, ndarray
import numpy as np

np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=10)


def univ_repr(qmc_object, abc_class_name, attributes):
    """
    Clean way to represent qmc_object data.

    Args:
        qmc_object (object): an qmc_object instance
        abc_class_name (str): name of the abstract class
        attributes (list): list of attributes to include

    Returns:
        str: string representation of this qmcpy object

    Note:
        print(qmc_object) is equivalent to print(qmc_object.__repr__()). 
        See an abstract classes __repr__ method for example call to this method. 
    """
    unique_attributes = []
    for attrib in attributes:
        if attrib not in unique_attributes:
            unique_attributes += [attrib]
    string = "%s (%s Object)\n" % (type(qmc_object).__name__, abc_class_name)
    for key in unique_attributes:
        val = getattr(qmc_object, key)
        # list of one value becomes just that value
        if isinstance(val, list) and len(val) == 1:
            val = val[0]
        elif isinstance(val, list):
            val = array(val)
        elif isinstance(val, ndarray):
            if val.shape == (1,):
                val = val[0].item()
            elif val.shape == ():
                val = val.item()
        # printing options
        if isinstance(val, int) or (isinstance(val, float) and val % 1 == 0):
            string_temp = '\t%-15s %d' % (key, int(val))
        elif isinstance(val, float):
            string_temp = '\t%-15s %0.4f' % (key, val)
        else:
            string_temp = '\t%-15s %s' % (key, val)
        string += string_temp.replace('\n', '\n\t%-15s' % ' ') + '\n'
    return string
