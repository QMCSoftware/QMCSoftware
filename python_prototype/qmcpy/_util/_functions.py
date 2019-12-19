""" Utility functions. Not meant for public use """

from numpy import array, int64, ndarray, repeat
import numpy as np

from . import DimensionError

np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=3)


def univ_repr(qmc_object, abc_class_name, attributes):
    """Clean way to represent qmc_object data.

    Note: ::

        print(qmc_object)

    is equivalent to ::tes)

        print(qmc_object.__repr__())

    Args:
        qmc_object (object): an qmc_object instance
        abc_class_name (str): name of the abstract class
        attributes (list): list of attributes to include

    Returns:
        str

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
            string_temp = '\t%-15s %0.3f' % (key, val)
        else:
            string_temp = '\t%-15s %s' % (key, val)
        string += string_temp.replace('\n', '\n\t%-15s' % ' ') + '\n'
    return string


def multilevel_constructor(self, dimension, **kwargs):
    """
    Takes an instance (self) and copies it into a list of instances (of self)
    with keyword arguments (kwargs) distributed to list instances

    Args:
        self (object): instance of the object
        dimension (int / list / ndarray): dimension of each level. len(dimension) == # levels
        **kwargs (tuple): keyword arguments

    Return:
        obj_list (list): a list of objects of type(self) with args and keyword
                         arguments distributed accordingly
    """
    # Type check dimension
    if isinstance(dimension, (int, int64)):
        # int -> ndarray
        dimension = array([dimension])
    if all(isinstance(i, (int, int64)) and i > 0 for i in dimension):
        # all positive integers
        dimension = array(dimension)
    else:
        raise DimensionError(
            "dimension must be an numpy.ndarray/list of positive integers")
    # Constants
    keys = list(kwargs.keys())
    n_levels = len(dimension)
    _type = type(self)
    # Type check measure data
    for key in keys:
        try:
            if len(kwargs[key]) == n_levels:  # already correctly formatted
                continue
        except:
            pass
        kwargs[key] = repeat(kwargs[key], n_levels)
    # Give the construcing object the correctly formatted measure data
    for key in keys:
        setattr(self, key, kwargs[key])
    setattr(self, 'dimension', dimension)  # set dimension attribute for self
    # Create a list of measures and distribute measure data
    obj_list = [_type.__new__(_type) for i in range(n_levels)]
    # ith object gets ith value from each measure data
    for i in range(n_levels):
        obj_list[i].dimension = dimension[i]
        for key, val in kwargs.items():
            setattr(obj_list[i], key, val[i])
    return obj_list
