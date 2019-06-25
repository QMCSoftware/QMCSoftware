''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from abc import ABC, abstractmethod
from numpy import sqrt,cumsum,diff,insert
from util import univ_repr
from scipy.stats import norm

class fun(ABC):
    '''
    Specify and generate values $f(\vx)$ for $\vx \in \cx$
        Any sublcass of fun must include:
            Methods: g(self,x,coordIndex)
    '''
    
    def __init__(self,nominalValue=None):
        super().__init__()
        self.nominalValue = nominalValue if nominalValue else 0  # a nominal number, $c$, such that $(c, \ldots, c) \in \cx$
        self.f = None # function handle of integrand after transformation
        self.dimension = 2  # dimension of the domain, $d$
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
            self[ii].dimension = dstrObj[ii].trueD.dimension # the function needs the dimension
            if msrObj[ii].measureName==dstrObj[ii].trueD.measureName:
                self[ii].f = lambda xu,coordIdex,i=ii: self[i].g(xu,coordIdex)
            elif msrObj[ii].measureName=='IIDZMeanGaussian' and dstrObj[ii].trueD.measureName=='stdGaussian': # multiply by the likelihood ratio
                this_var = msrObj[ii].measureData['variance']
                self[ii].f = lambda xu,coordIndex,var=this_var,i=ii: self[i].g(xu*sqrt(var),coordIndex)
            elif msrObj[ii].measureName=='BrownianMotion' and dstrObj[ii].trueD.measureName=='stdGaussian':
                timeDiff = diff(insert(msrObj[ii].measureData['timeVector'],0,0))
                self[ii].f = lambda xu,coordIndex,timeDiff=timeDiff,i=ii: self[i].g(cumsum(xu*sqrt(timeDiff),1),coordIndex)
            elif msrObj[ii].measureName=='IIDZMeanGaussian' and dstrObj[ii].trueD.measureName=='stdUniform':
                this_var = msrObj[ii].measureData['variance']
                self[ii].f = lambda xu,coordIdex,var=this_var,i=ii: self[i].g(sqrt(var)*norm.ppf(xu),coordIdex)
            elif msrObj[ii].measureName=='BrownianMotion' and dstrObj[ii].trueD.measureName=='stdUniform':
                raise Exception("Not yet implemented")
            else:
                raise Exception("Variable transformation not performed")
        return self

    # Magic Methods. Makes self[i]==self.fun_list[i]
    def __len__(self): return len(self.fun_list)   
    def __iter__(self):
        for fun in self.fun_list:
            yield fun
    def __getitem__(self,i): return self.fun_list[i]
    def __setitem__(self,i,val): self.fun_list[i] = val
    def __repr__(self): return univ_repr(self,'fun','fun_list')