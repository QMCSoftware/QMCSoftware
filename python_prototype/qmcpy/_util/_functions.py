""" Utility functions. Not meant for public use """

from . import DimensionError

from numpy import array, ndarray, tile, int64
import numpy as np

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
    obj_dict = qmc_object.__dict__
    string = "%s (%s Object)\n" % (type(qmc_object).__name__, abc_class_name)
    for key in unique_attributes:
        val = getattr(qmc_object, key)
        # list of one value becomes just that value
        if type(val) == list and len(val) == 1:
            val = val[0]
        elif type(val) == list:
            val = array(val)
        elif type(val) == ndarray:
            if val.shape == (1,):
                val = val[0].item()
            elif val.shape == ():
                val = val.item()
        # printing options
        if type(val) == int or (type(val) == float and val % 1 == 0):
            string_temp = '\t%-15s %d' % (key, int(val))
        elif type(val) == float:
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
        dimensions (int / list / ndarray): dimension of each level. len(dimension) == # levels
        **kwargs (tuple): keyword arguments
    
    Return:
        obj_list (list): a list of objects of type(self) with args and keyword
                         arguments distributed accordingly
    """
    # Type check dimension
    if isinstance(dimension, int): 
        # int -> ndarray
        dimension = array([dimension])
    if all(isinstance(i, (int, int64)) and i > 0 for i in dimension): 
        # all positive integers
        dimension = array(dimension)
    else:
        raise DimensionError(
            "dimension must be an numpy.ndarray/list of positive integers")
    # Type check measureData
    for key, val in kwargs.items():
        if not isinstance(kwargs[key], (list, ndarray)):
            # put single value into a list
            kwargs[key] = [kwargs[key]]
        if len(kwargs[key]) == 1 and len(dimension) != 1:
            # copy single-value to all levels
            kwargs[key] = tile(array(kwargs[key]), len(dimension))
        if len(kwargs[key]) != len(dimension):
            raise DimensionError(
                key + " must be a numpy.ndarray (or list) of len(dimension)")
        # properly specified, assign back to object
        setattr(self, key, val)
    setattr(self, 'dimension', dimension) # set dimension attribute for self
    # note: self must have dimension as its first argument to the constructor
    obj_list = [type(self)(None) for i in range(len(dimension))]
    # Create list of measures with proper dimensions and keyword arguments
    for i in range(len(obj_list)):
        obj_list[i].dimension = dimension[i]
        for key, val in kwargs.items():
            setattr(obj_list[i], key, array(val[i]))
    return obj_list