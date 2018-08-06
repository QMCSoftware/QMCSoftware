from abc import ABC, abstractmethod
 
class AbstractClassExample(ABC):
 
    def __init__(self, value):
        self.value = value
        super().__init__()
    
    @abstractmethod
    def do_something(self):
        pass

class DoAdd42(AbstractClassExample):
    def do_something(self):
        return self.value + 42
    
class DoMul42(AbstractClassExample):
    def __init__(self,val1):
        self.val1 = val1
        super().__init__(2*val1)
    def do_something(self):
        return self.val1 * + self.value
    
#x = DoAdd42(10)
y = DoMul42(10)
#print(x.do_something())
print(y.do_something())
print(y.val1)
print(y.value)