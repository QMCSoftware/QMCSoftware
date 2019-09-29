''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from abc import ABC, abstractmethod

from numpy import cumsum, diff, insert, sqrt
from scipy.stats import norm

from .. import univ_repr


class Fun(ABC):
    '''
    Specify and generate values $f(\vx)$ for $\vx \in \cx$
        Any sublcass of Fun must include:
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
    
    def transform_variable(self, msr_obj, dstr_obj):
        '''
        This method performs the necessary variable transformationto put the 
        original function in the form required by the discreteDistributon
        object starting from the original Measure object

        msr_obj = the Measure object that defines the integral
        dstr_obj = the discrete distribution object that is sampled from
        '''
        for ii in range(len(self)):
            self[ii].dimension = dstr_obj[ii].trueD.dimension # the function needs the dimension
            if msr_obj[ii].measureName==dstr_obj[ii].trueD.measureName:
                self[ii].f = lambda xu,coordIdex,i=ii: self[i].g(xu,coordIdex)
            elif msr_obj[ii].measureName== 'iid_zmean_gaussian' and dstr_obj[ii].trueD.measureName== 'std_gaussian': # multiply by the likelihood ratio
                this_var = msr_obj[ii].measureData['variance']
                self[ii].f = lambda xu,coordIndex,var=this_var,i=ii: self[i].g(xu*sqrt(var),coordIndex)
            elif msr_obj[ii].measureName== 'brownian_motion' and dstr_obj[ii].trueD.measureName== 'std_gaussian':
                timeDiff = diff(insert(msr_obj[ii].measureData['timeVector'], 0, 0))
                self[ii].f = lambda xu,coordIndex,timeDiff=timeDiff,i=ii: self[i].g(cumsum(xu*sqrt(timeDiff),1),coordIndex)
            elif msr_obj[ii].measureName== 'iid_zmean_gaussian' and dstr_obj[ii].trueD.measureName== 'std_uniform':
                this_var = msr_obj[ii].measureData['variance']
                self[ii].f = lambda xu,coordIdex,var=this_var,i=ii: self[i].g(sqrt(var)*norm.ppf(xu),coordIdex)
            elif msr_obj[ii].measureName== 'brownian_motion' and dstr_obj[ii].trueD.measureName== 'std_uniform':
                timeDiff = diff(insert(msr_obj[ii].measureData['timeVector'], 0, 0))
                self[ii].f = lambda xu,coordIndex,timeDiff=timeDiff,i=ii: self[i].g(cumsum(norm.ppf(xu)*sqrt(timeDiff),1),coordIndex)
            else:
                raise Exception("Variable transformation not performed")
        return self

    # Magic Methods. Makes self[i]==self.fun_list[i]
    def __len__(self): return len(self.fun_list)   
    def __iter__(self):
        for fun in self.fun_list: yield fun
    def __getitem__(self,i): return self.fun_list[i]
    def __setitem__(self,i,val): self.fun_list[i] = val
    def __repr__(self): return univ_repr(self,'fun_list')