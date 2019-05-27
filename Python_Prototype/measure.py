from numpy import array,arange

class measure():
    '''
    Specifies the components of a general measure used to define an
    integration problem or a sampling method
    '''
    def __init__(self,dimension=2,domainCoord=None,domainShape=None,measureName=None,
                measureData=None,variance=None,timeVector=None,_constructor=True):  
        
        self.measureName = measureName # name of the measure
        self.dimension = dimension if type(dimension)!=int else [dimension]# dimension of the domain, Â§$\mcommentfont d$Â§
        self.domainCoord = domainCoord # domain coordinates for the measure, Â§$\mcommentfont \cx$Â§
        self.domainShape = domainShape # domain shape for the measure, Â§$\mcommentfont \cx$Â§
        self.measureName = measureName if measureName else 'stdUniform'
        self.measureData = measureData # information required to specify the measure
        self.variance = variance
        self.timeVector = timeVector
        self.measure_list = []
        
        # Argument Parsing
        acceptedMeasures = {
            'stdUniform': self.stdUniform,
            'uniform': None,
            'stdGaussian': self.stdGaussian,
            'IIDZMeanGaussian': self.IIDZMeanGaussian,
            'IIDGaussian': None,
            'BrownianMotion': self.BrownianMotion,
            'Gaussian': None,
            'Lesbesgue': None}

        # Construct list of measure objects
        if _constructor:
            nObj = len(self.dimension) if not self.timeVector else len(self.timeVector)
            self.measure_list = [measure(_constructor=False) for i in range(nObj)] # implicitly referenced with self using magic methods
            # Deal attributes/defaults to remaining objects
            for i in range(len(self)):
                self[i].domainCoord = self.domainCoord[i] if self.domainCoord else array([])
                self[i].domainShape = self.domainShape[i] if self.domainShape else ''
                self[i].measureData = self.measureData[i] if self.measureData else array([])
            acceptedMeasures[self.measureName]()
            # TypeCheck
            for measureObj in self: 
                measureObj.typeCheck(acceptedMeasures)
    
    def typeCheck(self,acceptedMeasures):
        '''
        Check DataTypes of measure objects in self.measure_list
        Will only run all measures objects in self.measure_list are completely constructred
        '''
        if self.measureName not in acceptedMeasures.keys():
            raise Exception('self.measureName must consist of: '+str(list(acceptedMeasures.keys())))
        if type(self.dimension)!=int or self.dimension<=0:
            raise Exception("measure.dimension be a list of positive integers")
        if self.domainShape not in ['','box','cube','unitCube']:
            raise Exception("measure.domainShape must consist of: ['','box','cube','unitCube']")
        
    def stdUniform(self):
        ''' create standard uniform measure '''
        for i in range(len(self)):
            self[i].dimension = self.dimension[i]
            self[i].measureName = 'stdUniform'
        return self

    def stdGaussian(self):
        ''' create standard Gaussian measure '''
        for i in range(len(self)):
            self[i].dimension = self.dimension[i]
            self[i].measureName = 'stdGaussian'
        return self
    
    def IIDZMeanGaussian(self):
        ''' create standard Gaussian measure '''
        for i in range(len(self)):
            self[i].dimension = self.dimension[i]
            self[i].variance = self.variance[i]
            self[i].measureName = 'IIDZMeanGaussian'
        return self
    
    def BrownianMotion(self):
        ''' create a discretized Brownian Motion measure '''
        self.dimension = []
        for i in range(len(self)):
            self[i].timeVector = self.timeVector[i]
            dim_i = len(self[i].timeVector)
            self[i].dimension = dim_i
            self.dimension.append(dim_i)
            self[i].measureName = 'BrownianMotion'
        return self
        
    # Below methods allow the measure class to be treated like a list of measures
    def __len__(self):
        return len(self.measure_list)
    def __iter__(self):
        for measureObj in self.measure_list:
            yield measureObj
    def __getitem__(self,i):
        return self.measure_list[i]
    
    def __repr__(self):
        s = str(type(self).__name__)+' with properties:\n'
        for key,val in self.__dict__.items():
                s += '    %s: %s\n'%(str(key),str(val))
        return s[:-1]
        
if __name__ == "__main__":
    # Doctests
    import doctest
    x = doctest.testfile("Tests/dt_measure.py")
    print("\n"+str(x))
        