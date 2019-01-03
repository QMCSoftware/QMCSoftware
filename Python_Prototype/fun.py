from abc import ABC, abstractmethod
from numpy import sqrt

# Specify and generate values $f(\vx)$ for $\vx \in \cx$
class fun(ABC): # In the transform commit 'fun' is no longer Abstract. I'm not sure why. 
    '''
    Any sublcass of fun must include:
        Properties: None
        Methods: f(self, x, coordIndex) 
    '''
    funObjs = [] # Class variable of all instances. Need to find a way to reinitialize this to 0 whenever a new problem is started
    def __init__(self):
        super().__init__()
        self.domain = None # domain of the function, $\cx$
        self.domainType = 'box'  # e.g., 'box', 'ball'
        self.dimension = 2  # dimension of the domain, $d$
        self.distrib = {'name':'stdUniform'}  # e.g., 'uniform', 'Gaussian', 'Lebesgue'
        self.nominalValue = 0  # a nominal number, $c$, such that $(c, \ldots, c) \in \cx$
        fun.funObjs.append(self) # adds the function object to the list. Would be used with matlab equivalent variable transform method
    
    # Below methods allow the fun class to be treated like a special instance of the list class
    def __len__(self):
        return len(fun.funObjs)
    def __iter__(self):
        for funObj in fun.funObjs:
            yield funObj
    def __getitem__(self, i):
        return fun.funObjs[i]
    
    # Abstract Methods
    @abstractmethod
    def g(self, x, coordIndex):
        '''
         x = nodes, $\vx_{\fu,i} = i^{\text{th}}$ row of an $n \times |\fu|$ matrix
         coordIndex = set of those coordinates in sequence needed, $\fu$
         y = $n \times p$ matrix with values $f(\vx_{\fu,i},\vc)$ where
            if $\vx_i' = (x_{i,\fu},\vc)_j$, then $x'_{ij} = x_{ij}$ for $j \in \fu$, and $x'_{ij} = c$ otherwise
        '''
        pass
    
    # Possible variable transofrmation method equivalent (mostly) to the matlab implementation
    # What happens in the case of a keister function where neither the if or elif is defined? Then the fuctction f is never defined
    def transformVariable(self,dstrObj): # INCOMPLETE
        # This method performs the necessary variable transformation to put the original function in the form required by the discreteDistributonobject
        # Would fail if we create many fun instances across multiple problems. Would transform all instances instead of just those relevent to the problem 
        for ii in range(len(fun.funObjs)):  # iterates through all instances of the fun object
            if fun.funObjs[ii].distrib['name'] == dstrObj.trueDistribution: 
                fun.funObjs[ii].f = lambda xu,coordIndex: fun.funObjs[ii].g(xu, coordIndex) # MAY NOT BE RIGHT
            elif fun.funObjs[ii].distrib['name'] == 'IIDZGaussian' and dstrObj.trueDistribution =='stdGaussian':  # multiply by the likelihood ratio
                fun.funObjs[ii].f = lambda xu,coordIndex: fun.funObjs[ii].g(xu*sqrt(fun.funObjs[ii].distrib['variance']), coordIndex)
        return self
    
    '''
    MATLAB EQUIVALENT
    function obj = transformVariable(obj,dstrObj)
        %This method performs the necessary variable transformation to put the
        %original function in the form required by the discreteDistributon
        %object
        for ii = 1:numel(obj)
            if strcmp(obj(ii).distrib.name,dstrObj.trueDistribution)
                obj(ii).f = @(xu, coordIndex) g(obj(ii), xu, coordIndex);
            elseif strcmp(obj(ii).distrib.name,'IIDZGaussian') && ... 
                    strcmp(dstrObj.trueDistribution,'stdGaussian') %multiply by the likelihood ratio
                obj(ii).f = @(xu, coordIndex) g(obj(ii), xu*sqrt(obj.distrib.variance), ...
                    coordIndex);
            end
        end
    end
    '''