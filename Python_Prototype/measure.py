from numpy import array

class measure:
    '''
    Specifies the components of a general measure used to define an
    integration problem or a sampling method
    '''
    measureObjs = []
    def __init__(self,dimension=2,domainCoord=array([]),domainShape='',measureName='stdUniform',measureData=array([])):  
        # Argument Parsing
        if type(dimension)!=int or dimension<=0:
            raise Exception("measure.dimension must be a positive integer")
        acceptedDomainShapes = ['','box','cube','unitCube']
        if domainShape not in acceptedDomainShapes:
            raise Exception("measure.domainShape must be one of:"+str(acceptedDomainShapes))
        acceptedMeasureNames = ['stdUniform','uniform','stdGaussian','IIDZMeanGaussian','IIDGaussian','BrownianMotion','Gaussian','Lesbesgue']
        if measureName not in acceptedMeasureNames:
            raise Exception('self.measureName must be one of:'+str(acceptedMeasureNames))
        self.dimension = dimension # dimension of the domain, Â§$\mcommentfont d$Â§
        self.domainCoord = domainCoord # domain coordinates for the measure, Â§$\mcommentfont \cx$Â§
        self.domainShape = domainShape # domain shape for the measure, Â§$\mcommentfont \cx$Â§
        self.measureName = measureName # name of the measure
        self.measureData = measureData # information required to specify the measure
    
    def stdUniform(self,dimension=2):
        ''' create standard uniform measure '''
        try: nObj = len(dimension)
        except: nObj = 1
        measure.measureObjs = [measure() for i in range(nObj)]
        self.dimension = dimension
        self.measureName = 'stdUniform'
        return self
    
    def stdGaussian(self,dimension=2):
        ''' create standard Gaussian measure '''
        try: nObj = len(dimension)
        except: nObj = 1
        measure.measureObjs = [measure() for i in range(nObj)]
        self.dimension = dimension
        self.measureName = 'stdGaussian'
        return self
    
    def IIDZMeanGaussian(self,dimension=2,variance=1):
        ''' create standard Gaussian measure '''
        try: nObj = len(dimension)
        except: nObj = 1
        measure.measureObjs = [measure() for i in range(nObj)]
        self.dimension = dimension
        self.measureData.variance = variance
        self.measureName = 'IIDZMeanGaussian'
        return self
    
    def BrownianMotion(self,timeVector=arange(1/4,5/4,1/4)):
        ''' create a discretized Brownian Motion measure '''
        nObj = len(timeVector)
        measure.measureObjs = [measure() for i in range(nObj)]
        for ii in range(nObj):
            self[ii].measureData.timeVector = timeVector[ii]
            self[ii].dimension = len(self[ii].measureData.timeVector)
        self.measureName = 'BrownianMotion'
        
    # Below methods allow the measure class to be treated like a list of measures
    def __len__(self):
        return len(measure.measureObjs)
    def __iter__(self):
        for measureObj in measure.measureObjs:
            yield measureObj
    def __getitem__(self,i):
        return measure.measureObjs[i]
        


        