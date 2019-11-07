""" Utility functions. Not meant for public use """
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=5)

def univ_repr(qmc_object, abc_class_name, attributes):
    """Clean way to represent qmc_object data.

    Note: ::

        print(qmc_object)

    is equivalent to ::

        print(qmc_object.__repr__())

    Args:
        qmc_object (object): an qmc_object instance
        abc_class_name (str): name of the abstract class
        attributes (list): list of attributes to include

    Returns:
        str

    """
    obj_dict = qmc_object.__dict__
    string = "%s (%s Object)\n" % (type(qmc_object).__name__,abc_class_name)
    for key, val in obj_dict.items():
        if key not in attributes:
            # don't care about this attribute
            continue
        # list of one value becomes just that value
        if type(val) == list and len(val) == 1:
            val = val[0]
        if type(val) == np.ndarray and val.shape == (1,):
            val = val[0].item()
        # printing options
        elif type(val) == float:
            string += '\t%-15s %0.3f\n' % (key,val)
        else:
            string += '\t%-15s %s\n' % (key,val)
    return string