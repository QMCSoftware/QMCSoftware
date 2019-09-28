''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''

def univ_repr(self,objList_s=None):
    '''
    clean way to represent object data
    note: print(obj) == print(obj.__repr__())
    '''
    s = '%s object with properties:\n'%(type(self).__name__)
    for key,val in self.__dict__.items():
        if str(key) != objList_s:
            s += '%4s%s: %s\n'%('',str(key),str(val).replace('\n','\n%15s'%('')))
    if not objList_s: return s[:-1]
    # print list of subObject with properties
    s += '    %s:\n'%(objList_s)
    for i,subObj in enumerate(self):
        s += '%8s%s[%d] with properties:\n'%('',objList_s,i)
        for key,val in subObj.__dict__.items():
            if str(key) != objList_s:
                s += '%12s%s: %s\n'%('',str(key),str(val).replace('\n','\n%20s'%('')))
    return s[:-1]