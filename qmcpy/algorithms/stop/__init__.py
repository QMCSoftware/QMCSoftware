''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from abc import ABC, abstractmethod

from .. import univ_repr

class stoppingCriterion(ABC):
    '''
    Decide when to stop
        Any sublcass of stoppingCriterion must include:
            Methods: stopYet(self,distribObj) 
            Properties: discDistAllowed, decompTypeAllowed  
    '''
    
    def __init__(self,discDistAllowed,decompTypeAllowed,absTol=None,relTol=None,nInit=None,nMax=None): 
        super().__init__()
        self.absTol = absTol if absTol else 1e-2 # absolute tolerance, Â§$\mcommentfont d$Â§
        self.relTol = relTol if relTol else 0 # relative tolerance, Â§$\mcommentfont d$Â§
        self.nInit = nInit if nInit else 1024 # initial sample size
        self.nMax = nMax if nMax else 1e8 # maximum number of samples allowed
        # Abstract Properties
        self.discDistAllowed = discDistAllowed # which discrete distributions are supported
        self.decompTypeAllowed = decompTypeAllowed # which decomposition types are supported
    
    # Abstract Methods
    @abstractmethod
    def stopYet(self,distribObj): # distribObj = data or summary of data computed already
        pass

    def __repr__(self): return univ_repr(self)