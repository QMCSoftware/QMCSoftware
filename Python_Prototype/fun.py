from abc import ABC, abstractmethod
from numpy import sqrt,cumsum
from util import univ_repr

class fun(ABC):
    ''' Specify and generate values $f(\vx)$ for $\vx \in \cx$ '''
    
    def __init__(self):
        super().__init__()
        self.f = None # function handle of integrand after transformation
        self.dimension = 2  # dimension of the domain, $d$
        self.nominalValue = 0  # a nominal number, $c$, such that $(c, \ldots, c) \in \cx$
        self.fun_list = [self]
    
    # Abstract Methods
    @abstractmethod
    def g(self, x, coordIndex): # original function to be integrated
        '''
        xu = nodes, Â§\mcommentfont $\vx_{\fu,i} = i^{\text{th}}$ row of an $n \times |\fu|$ matrixÂ§
        coordIndex = set of those coordinates in sequence needed, Â§\mcommentfont $\fu$Â§
        y = Â§\mcommentfont$n \times p$ matrix with values $f(\vx_{\fu,i},\vc)$ where if $\vx_i' = (x_{i,\fu},\vc)_j$, then $x'_{ij} = x_{ij}$ for $j \in \fu$, and $x'_{ij} = c$ otherwiseÂ§
        '''
        pass
    
    def transformVariable(self,msrObj,dstrObj): # INCOMPLETE
        '''
        This method performs the necessary variable transformationto put the 
        original function in the form required by the discreteDistributon
        object starting from the original measure object

        msrObj = the measure object that defines the integral
        dstrObj = the discrete distribution object that is sampled from
        '''
        for ii in range(len(self)):
            self[ii].dimension = dstrObj[ii].trueD.dimension # the function needs the dimension also
            if msrObj[ii].measureName==dstrObj[ii].trueD.measureName:
                self[ii].f = lambda xu,coordIdex: self[ii].g(xu,coordIdex)
            elif msrObj[ii].measureName=='IIDZMeanGaussian' and dstrObj[ii].trueD.measureName=='stdGaussian': # multiply by the likelihood ratio
                this_variance = msrObj[ii].measureData['variance']
                self[ii].f = lambda xu,coordIndex: self[ii].g(xu*(this_variance)**.5,coordIndex)
            elif msrObj[ii].measureName=='BrownianMotion' and dstrObj[ii].trueD.measureName=='stdGaussian':
                timeDiff = msrObj[ii].measureData['timeVector']
                self[ii].f = lambda xu,coordIndex: self[ii].g(cumsum(xu*(timeDiff)**.5,0),coordIndex)
            else:
                raise Exception("Variable transformation not performed")
        return self

    # Allow this to be treated as a list of functions
    def __len__(self):
        return len(self.fun_list)   
    def __iter__(self):
        for fun in self.fun_list:
            yield fun
    def __getitem__(self,i):
        return self.fun_list[i]
    
    def __repr__(self): return univ_repr(self,'fun','fun_list')