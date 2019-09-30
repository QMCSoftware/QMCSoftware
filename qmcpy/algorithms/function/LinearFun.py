from . import Fun

class LinearFun(Fun):
    
    def __init__(self, nominal_value=None):
        super().__init__(nominal_value=nominal_value)
        
    def g(self,x,coordIndex):
        y = x.sum(1)
        return y