def univ_repr(self, obj_list_s=None) -> str:
    """
    Clean way to represent object data.

    Note: print(obj) == print(obj.__repr__())

    Args:
        self: an object instance
        obj_list_s:

    Returns:
        str

    """
    s = '%s object with properties:\n' % (type(self).__name__)
    for key, val in self.__dict__.items():
        if str(key) != obj_list_s:
            s += '%4s%s: %s\n' % \
                 ('', str(key), str(val).replace('\n', '\n%15s' % ('')))
    if not obj_list_s:
        return s[:-1]
    # print list of subObject with properties
    s += '    %s:\n' % (obj_list_s)
    for i, subObj in enumerate(self):
        s += '%8s%s[%d] with properties:\n' % ('', obj_list_s, i)
        for key, val in subObj.__dict__.items():
            if str(key) != obj_list_s:
                s += '%12s%s: %s\n' % \
                     ('', str(key), str(val).replace('\n', '\n%20s' % ('')))
    return s[:-1]
