class A(object):
    def __init__(self):
        import numpy as np
        self.imported = np

class B(A):
    def __init__(self):
        super().__init__()
        a = self.imported.array([1,2])
        print(a)


if __name__ == "__main__":
    import sys
    import os
    print(sys.path)
    old_cwd = os.getcwd()
    new_cwd = old_cwd +"\Accumulate_Data"
    sys.path.insert(0, new_cwd)
    print("\n",sys.path)
    sys.path.insert(0, new_cwd)
    print("\n",sys.path)

    meanVar = __import__("meanVarData")
    meanVar = getattr(meanVar,"meanVarData")
    a = meanVar()
    print(a)
    sys.path.remove(new_cwd)
    