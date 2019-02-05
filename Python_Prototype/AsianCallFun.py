from fun import fun as fun
from numpy import square, cos, exp, sqrt, multiply, sum, array, cumprod, transpose, ones
from numpy.linalg import eig
from scipy.sparse import spdiags

class AsianCallFun(fun):

    # Specify and generate payoff values of an Asian Call option
    def __init__(self, dimFac = []):
        super().__init__()
        self.volatility = 0.5
        self.S0 = 30
        self.K = 25
        self.T = 1
        self.A = []
        self.tVec = []
        if dimFac == []:
            self.dimFac = 1
        else:
            self.dimVec = cumprod(dimFac, axis=0)
            nf = self.dimVec.size
            acf_array = [AsianCallFun() for i in range(nf)]
            acf_array[0].dimFac = 0
            for ii in range(nf):
                acf_array[ii].distrib['name'] = 'stdGaussian'
                d = self.dimVec(ii)
                if ii > 0:
                    acf_array[ii].dimFac = dimFac(ii - 1)
                acf_array[ii].dimension = d
                tvec = range(d) * (acf_array[ii].T / d)
                acf_array(ii).tVec = tvec
                CovMat = min(transpose(tvec),tvec)
                [eigVec, eigVal] = eig(CovMat, 'vector')
                acf_array[ii].A = multiply(sqrt(eigVal[-1:-1:1]), transpose(eigVec[:,-1:-1:1]))

            self = acf_array
        

    def g(self, x, coordIndex):
        # since the nominalValue = 0, this is efficient
        BM = multiply(x, self.A)
        SFine = self.S0 * exp((-self.volatility ^ 2 / 2) * self.tVec + self.volatility * BM)
        AvgFine = ((self.S0 / 2) + sum(SFine[:, 1:self.dimension-1], 2) + SFine[:, self.dimension] / 2) / self.dimension
        y = max(AvgFine - self.K, 0)
        if self.dimFac > 0:
            SCoarse = SFine[:, self.dimFac: self.dimFac:-1]
            dCoarse = self.dimension / self.dimFac
            AvgCoarse = ((self.S0 / 2) + sum(SCoarse[:, 1:dCoarse-1], 2) + SCoarse[:, dCoarse] / 2) / dCoarse
            y = y - max(AvgCoarse - self.K, 0)
        return y

# BREAK ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Aleksei's Partial Translation is Below
'''
from fun import fun as fun
import numpy as np

def cumprod_m(a):
    r,_ = a.shape
    if r != 1:
        return np.cumprod(a,axis = 0)
    return np.cumprod(a,axis = 1)

def min_m(a):
    b = np.eye(len(a))
    for i in range(len(a)):
        for j in range(len(a)):
            b[i,j] = min(a[i],a[j])
    return b



class AsianCall(fun): # Translated up to the comment "Left off here"
    def __init__(self,dimFac=None): # Can pass in dimFac as float, int, list, or numpy.ndarray. Will convert to numpy array regardless
        super().__init__()
        self.volatility = 0.5
        self.S0 = 30
        self.K = 25
        self.T = 1
        self.A = None
        self.tVec = None
        self.dimFac = dimFac

        # Handles self.dimFac
        if self.dimFac == None: # Not supplied by the user
            return 
        else:
            # Transforms self.dimFac to the correct type
            self.dimFac = np.array([1])
            if type(dimFac) == list:
                self.dimFac == np.asarray(dimFac).reshape(1,len(dimFac))
            elif dimFac == float or dimFac == int:
                self.dimFac = self.asArray([dimFac])
            else:
                raise Exception("AsianCall is constructed with a float, int, list, or (preferably) numpy.ndarray")

        self.dimVec = cumprod_m(self.dimFac)
        nf = len(dimVec)

        self.obj_list = list(range(nf))
        for x in range(nf):
            self.obj_list[x] = AsianCallFun()
        self.obj_list[0].dimFac = 0    
        
        for ii in range(nf):
            self.obj_list[ii].distrib = {'name':'stdGaussian'}
            d = dimVec[ii]
            if ii > 0:
                obj_list[ii].dimFac = dimFac[ii-1]
            obj_list[ii].dimension = d
            tvec = np.arange(1,1+d)*(obj[ii].T/d)
            self.obj_list[ii].tVec = tvec
                     
            # Left off here
            try:
                CovMat = min_g(tvec)
             
            except: # Try block should never throw exception with how min_g is created
                #CovMat = min(tvec'*ones(1,length(tvec)), ones(length(tvec),1)*tvec);
                raise Exception("Problem taking min(tvec',tvec). Exception not yet implemented")
            
            # Left off here
            [eigVec,eigVal] = eig(CovMat,'vector') # np.linalg.eig is not the sae as Matlab's eig function

            # Example of dissimilarity
            #    Matlab:
            #        a = (1:4)
            #        b = min(a',a)
            #        [t1,t2] = eig(b,'vector') 
            #    Python:
            #        import numpy as np
            #        a = np.array([[1,1,1,1],[1,2,2,2],[1,2,3,3],[1,2,3,4]])
            #        t1,t2 = np.linalg.eig(a)     
            
            try:
                self.obj_list[ii].A = np.sqrt(eigVal(end:-1:1)) .* eigVec(:,end:-1:1)';
            except:
                v = np.sqrt(eigVal(end:-1:1));
                n = len(v);
                self.obj_list[ii].A = (v * np.ones(1,n)) .* eigVec(:,end:-1:1)';

    def f(self, obj, x, coordIndex): # None of this function has been translated yet
        # Since the nominalValue = 0, this is efficient
        
        BM = x * obj.A
        try:
            SFine = obj.S0 * np.exp((-1* obj.volatility**2/2)*obj.tVec + obj.volatility * BM)
        except:
            n, ~ = BM.shape
            SFine = obj.S0*np.exp((-obj.volatility^2/2)*np.ones(n,1)*obj.tVec + obj.volatility * BM);
        AvgFine = ((obj.S0/2) + sum(SFine(:,1:obj.dimension-1),2) + SFine(:,obj.dimension)/2)/obj.dimension;
        y = max(AvgFine - obj.K,0);
        if obj.dimFac > 0
            SCoarse = SFine(:,obj.dimFac:obj.dimFac:end);
            dCoarse = obj.dimension/obj.dimFac;
            AvgCoarse = ((obj.S0/2) + sum(SCoarse(:,1:dCoarse-1),2) + SCoarse(:,dCoarse)/2)/dCoarse;
            y = y - max(AvgCoarse - obj.K,0);
        return y      
'''       

if __name__ == "__main__":
    # Run Doctests
    import doctest
    x = doctest.testfile("Tests/dt_AsianCallFun.py")
    print("\n" + str(x))

