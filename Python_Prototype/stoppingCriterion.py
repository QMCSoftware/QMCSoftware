from abc import ABC, abstractmethod

# Decide when to stop
class stoppingCriterion(ABC):
    '''
    Decide when to stop
    Any sublcass of stoppingCriterion must include:
        Properties: discDistAllowed, decompTypeAllowed
        Methods: stopYet(self,distribObj) 
    '''
    def __init__(self,discDistAllowed,decompTypeAllowed): 
        super().__init__()
        self.absTol = 1e-2 # absolute tolerance, Â§$\mcommentfont d$Â§
        self.relTol = 0 # relative tolerance, Â§$\mcommentfont d$Â§
        self.nInit = 1024 # initial sample size
        self.nMax = 1e8 # maximum number of samples allowed
        # Abstract Properties
        self.discDistAllowed = discDistAllowed # which discrete distributions are supported
        self.decompTypeAllowed = decompTypeAllowed # which decomposition types are supported
    
    # Abstract Methods
    @abstractmethod
    def stopYet(self,distribObj): # distribObj = data or summary of data computed already
        pass

    def __repr__(self):
        s = str(type(self).__name__)+' with properties:\n'
        for key,val in self.__dict__.items():
            s += '    %s: %s\n'%(str(key),str(val))
        return s[:-1]