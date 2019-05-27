from numpy import array,arange

class measure():
    '''
    Specifies the components of a general measure used to define an
    integration problem or a sampling method
    '''
    def __init__(self,\
        dimension=2,domainCoord=array([]),domainShape='',measureName='stdUniform',\
        measureData=array([]),variance=1,timeVector=arange(1/4,5/4,1/4),_constructor=True):  
        
        self.measureName = measureName # name of the measure
        self.dimension = dimension # dimension of the domain, Â§$\mcommentfont d$Â§
        self.domainCoord = domainCoord # domain coordinates for the measure, Â§$\mcommentfont \cx$Â§
        self.domainShape =domainShape # domain shape for the measure, Â§$\mcommentfont \cx$Â§
        self.measureData = measureData # information required to specify the measure
        self.variance = variance
        self.timeVector = timeVector
        self.measure_list = []

        if _constructor and type(dimension)==int:
            # Single Dimensional ==> Convert everything to lists so it appears multi-dimensional
            self.dimension = [dimension]
            self.domainCoord = [domainCoord]
            self.measureData = [measureData]
            self.variance = [variance]
            self.timeVector = [timeVector]
        
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
        if self.measureName not in acceptedMeasures.keys():
            raise Exception('self.measureName is the key for the function it will map to:'+str(acceptedMeasures))
        if not _constructor and (type(dimension)!=int or dimension<=0):
            raise Exception("measure.dimension be a list of positive integers")
        acceptedDomainShapes = ['','box','cube','unitCube']
        if self.domainShape not in acceptedDomainShapes:
            raise Exception("measure.domainShape must be one of:"+str(acceptedDomainShapes))
        if not _constructor and self.variance <=0:
            raise Exception("measure.variance must be a positive number")
        
        # Construct list of measure objects
        if _constructor:
            acceptedMeasures[measureName]()
        
    def stdUniform(self):
        ''' create standard uniform measure '''
        nObj = len(self.dimension)
        self.measure_list = list(range(nObj))
        for i in range(nObj):
            self.measure_list[i] = measure(dimension=self.dimension[i],domainCoord=self.domainCoord[i],domainShape=self.domainShape,
                                        measureName=self.measureName,measureData=self.measureData,variance=self.variance,
                                        timeVector=self.timeVector,_constructor=False)
        return self
    
    def stdGaussian(self,dimension=2):
        ''' create standard Gaussian measure '''
        nObj = len(self.dimension)
        self.measure_list = list(range(nObj))
        for i in range(nObj):
            self.measure_list[i] = measure(dimension=self.dimension[i],domainCoord=self.domainCoord[i],domainShape=self.domainShape,
                                        measureName=self.measureName,measureData=self.measureData,variance=self.variance,
                                        timeVector=self.timeVector,_constructor=False)
        return self
    
    def IIDZMeanGaussian(self,dimension=2,variance=1):
        ''' create standard Gaussian measure '''
        nObj = len(self.dimension)
        self.measure_list = list(range(nObj))
        for i in range(nObj):
            self.measure_list[i] = measure(dimension=self.dimension[i],domainCoord=self.domainCoord[i],domainShape=self.domainShape,
                                        measureName=self.measureName,measureData=self.measureData,variance=self.variance,
                                        timeVector=self.timeVector,_constructor=False)
        return self
    
    def BrownianMotion(self,timeVector):
        ''' create a discretized Brownian Motion measure '''
        nObj = len(timeVector)
        self.measure_list = list(range(nObj))
        for i in range(nObj):
            self.measure_list[i] = measure(dimension=len(self.timeVector[i]),domainCoord=self.domainCoord[i],domainShape=self.domainShape,
                                        measureName=self.measureName,measureData=self.measureData,variance=self.variance,
                                        timeVector=self.timeVector,_constructor=False)
        return self
        
    # Below methods allow the measure class to be treated like a list of measures
    def __len__(self):
        return len(self.measure_list)
    def __iter__(self):
        for measureObj in self.measure_list:
            yield measureObj
    def __getitem__(self,i):
        return self.measure_list[i]
        


        