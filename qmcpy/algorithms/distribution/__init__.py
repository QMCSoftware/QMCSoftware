''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from abc import ABC, abstractmethod
from numpy import array, arange

from .. import univ_repr

class MeasureCompatibilityError(Exception): pass

class DiscreteDistribution(ABC):
    '''
    Specifies and generates the components of $ a_n \sum_{i=1}^n w_i \delta_{\vx_i}(\cdot)$
        Any sublcass of DiscreteDistribution must include:
            Methods: gen_distrib(self,nStart,nEnd,n,coordIndex)
            Properties: distrib_data,trueD
    '''

    def __init__(self, accepted_measures, trueD=None, distrib_data=None):
        super().__init__()
        # Abstract Properties
        self.distrib_data = distrib_data
        self.trueD = trueD   
        self.distrib_list = [self]
        
        # Create self.distrib_list (self) and distribute attributes
        if trueD:
            if trueD.measureName not in accepted_measures:
                raise MeasureCompatibilityError(type(self).__name__+' only accepts measures:'+str(accepted_measures))
            self.distrib_list = [type(self)() for i in range(len(trueD))]
            for i in range(len(self)):    
                self[i].trueD = self.trueD[i]
                self[i].distrib_data = self.distrib_data[i] if self.distrib_data else None

    # Abstract Methods
    @abstractmethod
    def gen_distrib(self, n, m, j):
        """
         nStart = starting value of $ i$
         nEnd = ending value of $ i$
         n = value of $ n$ used to determine $ a_n$
         coordIndex = which coordinates in sequence are needed
        """
        pass
    
    # Magic Methods. Makes self[i]==self.distrib_list[i]
    def __len__(self): return len(self.distrib_list)
    def __iter__(self):
        for distribObj in self.distrib_list:
            yield distribObj
    def __getitem__(self,i): return self.distrib_list[i]
    def __setitem__(self,i,val): self.distrib_list[i]=val
    def __repr__(self): return univ_repr(self,'distrib_list')


class Measure():
    '''
    Specifies the components of a general Measure used to define an
    integration problem or a sampling method
    '''
    
    def __init__(self,domainShape='',domainCoord=None,measureData=None):  
        self.domainCoord = array(domainCoord) if domainCoord else array([])
        self.measureData = measureData if measureData else {}
        self.measure_list = []

    ''' Methods to construct list of measures ''' 
    def std_uniform(self, dimension=array([2])):
        ''' create standard uniform Measure '''
        self.measureName = 'std_uniform'
        #    Dimension of the domain, $ d$
        if type(dimension)==int: self.dimension = array([dimension])
        elif all(item>0 for item in dimension): self.dimension = array(dimension)
        else: raise Exception("Measure.dimension must be a list of positive integers")
        #    Construct list of measures
        self.measure_list = list(range(len(dimension)))
        for i in range(len(self.measure_list)):
            self.measure_list[i] = Measure()
            self.measure_list[i].dimension = self.dimension[i]
            self.measure_list[i].measureName = 'std_uniform'
        return self

    def std_gaussian(self, dimension=array([2])):
        ''' create standard Gaussian Measure '''
        self.measureName = 'std_gaussian'
        #    Dimension of the domain, $ d$
        if type(dimension)==int: self.dimension = array([dimension])
        elif all(item>0 for item in dimension): self.dimension = array(dimension)
        else: raise Exception("Measure.dimension be a list of positive integers")
        #    Construct list of measures
        self.measure_list = list(range(len(dimension)))
        for i in range(len(self.measure_list)):
            self.measure_list[i] = Measure()
            self.measure_list[i].dimension = self.dimension[i]
            self.measure_list[i].measureName = 'std_gaussian'
        return self

    def iid_zmean_gaussian(self, dimension=array([2]), variance=array([1])):
        ''' create standard Gaussian Measure '''
        self.measureName = 'iid_zmean_gaussian'
         #    Dimension of the domain, $ d$
        if type(dimension)==int: self.dimension = array([dimension])
        elif all(item>0 for item in dimension): self.dimension = array(dimension)
        else: raise Exception("Measure.dimension be a list of positive integers")
        #    Variance of Gaussian Measures
        if type(variance)==int: variance = array([variance])
        elif all(item>0 for item in variance): variance = array(variance)
        else: raise Exception("Measure.variance be a list of positive integers")
        #    Construct list of measures
        self.measure_list = list(range(len(dimension)))
        for i in range(len(self.measure_list)):
            self.measure_list[i] = Measure()
            self.measure_list[i].dimension = self.dimension[i]
            self.measure_list[i].measureData['variance'] = variance[i]
            self.measure_list[i].measureName = 'iid_zmean_gaussian'
        return self

    def brownian_motion(self, timeVector=arange(1 / 4, 5 / 4, 1 / 4)):
        ''' create a discretized Brownian Motion Measure '''
        self.measureName = 'brownian_motion'
        #    Dimension of domain, $ d$
        self.dimension = array([len(tV) for tV in timeVector])
        #    Construct list of measures
        self.measure_list = list(range(len(timeVector)))
        for i in range(len(self.measure_list)): 
            self.measure_list[i] = Measure()
            self.measure_list[i].measureData['timeVector'] = array(timeVector[i])
            self.measure_list[i].dimension = self.dimension[i]
            self.measure_list[i].measureName = 'brownian_motion'
        return self

    def lattice(self, dimension=array([2])):
        ''' low descrepancy lattice '''
        self.measureName = 'lattice'
        #    Dimension of the domain, $ d$
        if type(dimension)==int: self.dimension = array([dimension])
        elif all(item>0 for item in dimension): self.dimension = array(dimension)
        else: raise Exception("Measure.dimension be a list of positive integers")
        #    Construct list of measures
        self.measure_list = list(range(len(dimension)))
        for i in range(len(self.measure_list)):
            self.measure_list[i] = Measure()
            self.measure_list[i].measureData['lds_type'] = 'lattice'
            self.measure_list[i].dimension = self.dimension[i]
            self.measure_list[i].measureName = 'std_uniform'
        return self

    def sobol(self, dimension=array([2])):
        ''' low descrepancy sobol '''
        self.measureName = 'sobol'
        #    Dimension of the domain, $ d$
        if type(dimension)==int: self.dimension = array([dimension])
        elif all(item>0 for item in dimension): self.dimension = array(dimension)
        else: raise Exception("Measure.dimension be a list of positive integers")
        #    Construct list of measures
        self.measure_list = list(range(len(dimension)))
        for i in range(len(self.measure_list)):
            self.measure_list[i] = Measure()
            self.measure_list[i].measureData['lds_type'] = 'sobol'
            self.measure_list[i].dimension = self.dimension[i]
            self.measure_list[i].measureName = 'std_uniform'
        return self

    # Magic Methods. Makes self[i]==self.measure_list[i]
    def __len__(self): return len(self.measure_list)
    def __iter__(self):
        for measureObj in self.measure_list:
            yield measureObj
    def __getitem__(self,i): return self.measure_list[i]
    def __setitem__(self,i,val): self.measure_list[i] = val
    def __repr__(self): return univ_repr(self,'measure_list')

