from algorithms.function.fun import fun

class LinearFun(fun):
    
    def __init__(self,nominalValue=None):
        super().__init__(nominalValue=nominalValue)
        
    def g(self,x,coordIndex):
        y = x.sum(1)
        return y