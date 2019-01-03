from abc import ABC, abstractmethod

# Decide when to stop
class stoppingCriterion(ABC):
    '''
    Any sublcass of stoppingCriterion must include:
        Properties: discDistAllowed, decompTypeAllowed
        Methods: stopYet(self,distribObj) 
    '''
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
    
    # Abstract Methods
    @abstractmethod
    def stopYet(self,dataObj,funObj,distribObj): # distribObj = data or summary of data computed already
        pass