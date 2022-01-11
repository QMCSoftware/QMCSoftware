from ._accumulate_data import AccumulateData
from ..integrand._integrand import Integrand
from ..util import CubatureWarning
from numpy import *
import warnings


class LDTransformData(AccumulateData):
    """
    Update and store transformation data based on low-discrepancy sequences. 
    See the stopping criterion that utilize this object for references.
    """

    def __init__(self, m_min, coefv, fudge, check_cone, ncv, cv_mu, update_beta):
        self.m_min = m_min
        self.coefv = coefv
        self.fudge = fudge
        self.check_cone = check_cone
        self.ncv = ncv
        self.cv_mu = cv_mu
        self.update_beta = update_beta
        self.omg_circ = lambda m: 2**(-m)
        self.r_lag = 4
        self.l_star = self.m_min-self.r_lag
        self.omg_hat = lambda m: self.fudge(m)/((1+self.fudge(self.r_lag))*self.omg_circ(self.r_lag))
        self.stilde = 0
        self.c_stilde_low = tile(-inf,int(self.m_max-self.l_star+1))
        self.c_stilde_up = tile(inf,int(self.m_max-self.l_star+1))
        self.y_val = zeros(0,dtype=float)
        self.y_cp = zeros(0)
        self.yg_val = zeros((0,self.ncv),dtype=float)
        self.yg_cp = zeros((0,self.ncv))
        super(LDTransformData,self).__init__()

    def update_data(self, m, y_val_next, y_cp_next, yg_val_next, yg_cp_next):
        self.y_val = hstack((self.y_val,y_val_next))
        self.y_cp = hstack((self.y_cp,y_cp_next))
        self.yg_val = vstack((self.yg_val,yg_val_next))
        self.yg_cp = vstack((self.yg_cp,yg_cp_next))
        # fast transform
        for l in range(int(m)):
            nl = 2**l
            nmminlm1 = 2**(m-l-1)
            ptind_nl = hstack((tile(True,nl),tile(False,nl)))
            ptind = tile(ptind_nl,int(nmminlm1))
            coef = self.coefv(nl)
            coefv = tile(coef,int(nmminlm1))
            evenval = self.y_cp[ptind]
            oddval = self.y_cp[~ptind]
            self.y_cp[ptind] = (evenval+coefv*oddval)/2.
            self.y_cp[~ptind] = (evenval-coefv*oddval)/2.
            if self.ncv>0:
                evenval = self.yg_cp[ptind]
                oddval = self.yg_cp[~ptind]
                self.yg_cp[ptind] = (evenval+coefv[:,None]*oddval)/2.
                self.yg_cp[~ptind] = (evenval-coefv[:,None]*oddval)/2.


 



 
        ## Compute Stilde
        nllstart = int(2**(self.m-self.r_lag-1))
        self.stilde = sum(abs(self.y[self.kappanumap[nllstart:2*nllstart]-1]))
        ## Approximate integral
        self.solution = self.yval.mean()
        if self.ncv>0: # using control variates
            self.solution += self.beta@self.cv_mu # compensate for subtraction factor
        # update total samples
        self.n_total = 2**self.m # updated the total evaluations
        # Necessary conditions
        if not self.check_cone: return # don't check if the function falls in the cone
        for l in range(int(self.l_star),int(self.m+1)): # Storing the information for the necessary conditions
            c_tmp = self.omg_hat(self.m-l)*self.omg_circ(self.m-l)
            c_low = 1./(1+c_tmp)
            c_up = 1./(1-c_tmp)
            const1 = sum(abs(self.y[self.kappanumap[int(2**(l-1)):int(2**l)]-1]))
            idx = int(l-self.l_star)
            self.c_stilde_low[idx] = max(self.c_stilde_low[idx],c_low*const1)
            if c_tmp < 1:
                self.c_stilde_up[idx] = min(self.c_stilde_up[idx],c_up*const1)
        if (self.c_stilde_low > self.c_stilde_up).any():
            warnings.warn('An element of c_stilde_low > c_stilde_up, this function may violate the cone function. ', CubatureWarning)
        
    def _sub_update_data_first(self): 
        n = int(2**self.m_min)
        
       
        # create kappanumap from the data
        kappanumap = arange(1,n+1,dtype=int)
        for l in range(int(self.m_min-1),0,-1):
            nl = 2**l
            oldone = abs(y[kappanumap[1:int(nl)]-1]) # earlier values of kappa, don't touch first one
            newone = abs(y[kappanumap[nl+1:2*nl]-1]) # later values of kappa,
            flip = where(newone>oldone)[0]+1 # which in the pair are the larger ones. change to matlab indexing
            if flip.size != 0:
                additive = arange(0,2**self.m-1,2**(l+1)).reshape((1,-1))
                flipall = (flip.reshape((-1,1)) + additive)
                flipall = flipall.flatten('F').astype(int) # flatten column wise
                temp = kappanumap[nl+flipall] # then flip
                kappanumap[nl+flipall] = kappanumap[flipall] # them
                kappanumap[flipall] = temp # around   
        # if using control variates, find optimal beta
        if self.ncv>0:
            kappa_approx = kappanumap[int(2**(self.m_min-self.r_lag-1)):]-1 # kappa index used for fitting
            x4beta = yg[kappa_approx]
            y4beta = y[kappa_approx]
            self.beta = linalg.lstsq(x4beta,y4beta,rcond=None)[0]
            yval = ycv[:,0] - ycv[:,1:]@self.beta # get new function values
            y = y-yg@self.beta # redefine function
            # rebuild kappa map
            kappanumap = arange(1,n+1,dtype=int) # reinitialize
            for l in range(int(self.m_min-1),0,-1):
                nl = 2**l
                oldone = abs(y[kappanumap[1:int(nl)]-1]) # earlier values of kappa, don't touch first one
                newone = abs(y[kappanumap[nl+1:2*nl]-1]) # later values of kappa,
                flip = where(newone>oldone)[0]+1 # which in the pair are the larger ones. change to matlab indexing
                if flip.size != 0:
                    additive = arange(0,2**self.m-1,2**(l+1)).reshape((1,-1))
                    flipall = (flip.reshape((-1,1)) + additive)
                    flipall = flipall.flatten('F').astype(int) # flatten column wise
                    temp = kappanumap[nl+flipall] # then flip
                    kappanumap[nl+flipall] = kappanumap[flipall] # them
                    kappanumap[flipall] = temp # around
        # set some variables for the next iteration
        self.y,self.yval,self.kappanumap = y,yval,kappanumap
        if self.ncv>0: self.ycv = ycv

    def _sub_update_data_next(self):
        # any iteration after the first
        mnext = int(self.m-1)
        n = int(2**mnext)

        # compute fast transform
        if not self.update_beta: # do not update the beta coefficients
            # fast transform
            for l in range(mnext):
                nl = 2**l
                nmminlm1 = 2**(mnext-l-1)
                ptind_nl = hstack(( tile(True,nl), tile(False,nl) ))
                ptind = tile(ptind_nl,int(nmminlm1))
                coef = self.coefv(nl)
                coefv = tile(coef,int(nmminlm1))
                evenval = ynext[ptind]
                oddval = ynext[~ptind]
                ynext[ptind] = (evenval + coefv*oddval) / 2.
                ynext[~ptind] = (evenval - coefv*oddval) / 2.
            y = hstack((self.y,ynext))
            # compute fast transform on all points
            nl = 2**mnext
            ptind = hstack((tile(True,int(nl)),tile(False,int(nl))))
            coefv = self.coefv(nl)
            evenval = y[ptind]
            oddval = y[~ptind]
            y[ptind] = (evenval + coefv*oddval) / 2.
            y[~ptind] = (evenval - coefv*oddval) / 2.
            # update kappanumap
            kappanumap = hstack((self.kappanumap,int(2**(self.m-1))+self.kappanumap))
            for l in range(int(self.m-1),int(self.m-self.r_lag-1),-1):
                nl = 2**l
                oldone = abs(y[kappanumap[1:int(nl)]-1]) # earlier values of kappa, don't touch first one
                newone = abs(y[kappanumap[nl+1:2*nl]-1]) # later values of kappa,
                flip = where(newone>oldone)[0]+1 # which in the pair are the larger ones. change to matlab indexing
                if flip.size != 0:
                    additive = arange(0,2**self.m-1,2**(l+1)).reshape((1,-1))
                    flipall = (flip.reshape((-1,1)) + additive)
                    flipall = flipall.flatten('F').astype(int) # flatten column wise
                    temp = kappanumap[nl+flipall] # then flip
                    kappanumap[nl+flipall] = kappanumap[flipall] # them
                    kappanumap[flipall] = temp # around  
        else: # update beta
            ycv = vstack((self.ycv,ycvnext))
            y = ycv[:,0]
            yg = ycv[:,1:]
            if self.cast_complex: yg = yg.astype(complex)
            # compute fast transform 
            for l in range(int(self.m)):
                nl = 2**l
                nmminlm1 = 2**(self.m-l-1)
                ptind_nl = hstack(( tile(True,nl), tile(False,nl) ))
                ptind = tile(ptind_nl,int(nmminlm1))
                coef = self.coefv(nl)
                coefv = tile(coef,int(nmminlm1))
                evenval = y[ptind]
                oddval = y[~ptind]
                y[ptind] = (evenval + coefv*oddval) / 2.
                y[~ptind] = (evenval - coefv*oddval) / 2.
                evenval = yg[ptind]
                oddval = yg[~ptind]
                yg[ptind] = (evenval + coefv[:,None]*oddval) / 2.
                yg[~ptind] = (evenval - coefv[:,None]*oddval) / 2.
            # update beta approximation
            kappa_approx = self.kappanumap[int(2**(self.m-self.r_lag-1)):]-1 # kappa index used for fitting
            x4beta = yg[kappa_approx]
            y4beta = y[kappa_approx]
            self.beta = linalg.lstsq(x4beta,y4beta,rcond=None)[0]
            yval = ycv[:,0] - ycv[:,1:]@self.beta # get new function values
            y = y-yg@self.beta # redefine function
            # rebuild kappanumap
            kappanumap = hstack((self.kappanumap,2**(self.m-1)+self.kappanumap)).astype(int)
            for l in range(int(self.m-1),int(self.m-self.r_lag-1),-1):
                nl = 2**l
                oldone = abs(y[kappanumap[1:int(nl)]-1]) # earlier values of kappa, don't touch first one
                newone = abs(y[kappanumap[nl+1:2*nl]-1]) # later values of kappa,
                flip = where(newone>oldone)[0]+1 # which in the pair are the larger ones. change to matlab indexing
                if flip.size != 0:
                    additive = arange(0,2**self.m-1,2**(l+1)).reshape((1,-1))
                    flipall = (flip.reshape((-1,1)) + additive)
                    flipall = flipall.flatten('F').astype(int) # flatten column wise
                    temp = kappanumap[nl+flipall] # then flip
                    kappanumap[nl+flipall] = kappanumap[flipall] # them
                    kappanumap[flipall] = temp # around 
        # set some variables for the next iteration
        self.y,self.yval,self.kappanumap = y,yval,kappanumap
        if self.ncv>0 and self.update_beta: self.ycv = ycv
      