from abc import ABC, abstractmethod

# Decide when to stop
class Stopping_Criterion(ABC):

    def __init__(self): 
        self.absTol = 1e-2 # absolute tolerance 
        self.relTol = 0 # absolute tolerance
        self.nInit = 1024 # initial sample size
        self.nMax = 1e8 # maximum number of samples allowed
        super().__init__()

    # Abstract Properties 
    @property
    @abstractmethod
    def discDistAllowed(self): # which discrete distributions are supported
        pass
    
    @property
    @abstractmethod
    def decompTypeAllowed(self): # which decomposition types are supported
        pass
    
    # Abstract Method
    @abstractmethod
    def stopYet(self,dataObj,distribObj,funObj): # distribObj = data or summary of data computed already
        pass

'''
Any sublcass of stoppingCriterion must include:
    Methods: stopYet(self,distribObj)
    Properties: discDistAllowed, decompTypeAllowed
'''