from abc import ABC, abstractmethod
from numpy import sqrt,cumsum

# Specify and generate values $f(\vx)$ for $\vx \in \cx$
class fun(ABC): # In the transform commit 'fun' is no longer Abstract. I'm not sure why. 
    '''
    Any sublcass of fun must include:
        Properties: None
        Methods: f(self, x, coordIndex) 
    '''
    funObjs = [] # Class variable of all instances. Need to find a way to reinitialize this to [] whenever a new problem is started
    def __init__(self):
        super().__init__()
        self.f = None # function handle of integrand after transformation
        self.dimension = 2  # dimension of the domain, $d$
        self.nominalValue = 0  # a nominal number, $c$, such that $(c, \ldots, c) \in \cx$
        fun.funObjs.append(self) # adds the function object to the list
    
    # Abstract Methods
    @abstractmethod
    def g(self, x, coordIndex): # original function to be integrated
        '''
        % xu = nodes, Â§\mcommentfont $\vx_{\fu,i} = i^{\text{th}}$ row of an $n \times |\fu|$ matrixÂ§
        % coordIndex = set of those coordinates in sequence needed, Â§\mcommentfont $\fu$Â§
        % y = Â§\mcommentfont$n \times p$ matrix with values $f(\vx_{\fu,i},\vc)$ where if $\vx_i' = (x_{i,\fu},\vc)_j$, then $x'_{ij} = x_{ij}$ for $j \in \fu$, and $x'_{ij} = c$ otherwiseÂ§
        '''
        pass
    
    # Possible variable transofrmation method equivalent (mostly) to the matlab implementation
    # What happens in the case of a keister function where neither the if or elif is defined? Then the fuctction f is never defined?
    def transformVariable(self,msrObj,dstrObj): # INCOMPLETE
        # This method performs the necessary variable transformationto put the 
        # original function in the form required by the discreteDistributon
        # object starting from the original measure object

        # msrObj = the measure object that defines the integral
        # dstrObj = the discrete distribution object that is sampled from
        for ii in range(len(self)):
            self[ii].dimension = dstrObj[ii].trueD.dimension # the function needs the dimension also
            if msrObj[ii]==dstrObj[ii].trueD:
                self[ii].f = lambda xu,coordIdex: self[ii].g(xu,coordIdex)
            elif msrObj[ii].measureName=='IIDZMeanGaussian' and dstrObj[ii].trueD.measureName=='stdGaussian': # multiply by the likelihood ratio
                self[ii].f = lambda xu,coordIndex: self[ii].g(xu*(msrObj.measureData.variance)**.5,coordIndex)
            elif msrObj[ii].measureName=='BrownianMotion' and dstrObj[ii].trueD.measureName=='stdGaussian':
                timeDiff = msrObj(ii).measureData.timeVector
                self[ii].f = lambda xu,coordIndex: self[ii].g(cumsum(xu*(timeDiff)**.5,0),coordIndex)
            else:
                raise Exception("Variable transformation not performed")

    # Below methods allow the fun class to be treated like a list of functions
    def __len__(self):
        return len(fun.funObjs)
    def __iter__(self):
        for funObj in fun.funObjs:
            yield funObj
    def __getitem__(self,i):
        return fun.funObjs[i]