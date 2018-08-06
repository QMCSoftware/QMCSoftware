from abc import ABC, abstractmethod

# Decide when to stop
class stoppingCriterion(ABC):

    def __init__(self): 
        self.absTol = 1e-2 # absolute tolerance, Â§$\mcommentfont d$Â§
        self.relTol = 0 # absolute tolerance, Â§$\mcommentfont d$Â§
        self.nInit = 1024 # initial sample size
        self.nMax = 1e8 # maximum number of samples allowed

    # Abstract Properties 
    @property
    @abstractmethod
    def discDistAllowed(self): # which discrete distributions are supported
        pass
    
    @property
    @abstractmethod
    def decompTypeAllowed(self): # which decomposition types are supported
        pass
    
    #Abstract Method
    @abstractmethod
    def stopYet(self, distribObj): # distribObj = data or summary of data computed already
        pass

'''
Any sublcass of stoppingCriterion must include:
    Methods: stopYet(self,distribObj)
    Properties: discDistAllowed, decompTypeAllowed
'''