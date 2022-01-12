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

    def __init__(self, m_min, m_max, coefv, fudge, check_cone, ncv, cv_mu, update_beta):
        self.m_min = m_min
        self.m_max = m_max
        self.coefv = coefv
        self.fudge = fudge
        self.check_cone = check_cone
        self.ncv = ncv
        self.cv_mu = cv_mu
        self.update_beta = update_beta
        self.omg_circ = lambda m: 2**(-m)
        self.r_lag = 4
        self.l_star = int(self.m_min-self.r_lag)
        self.omg_hat = lambda m: self.fudge(m)/((1+self.fudge(self.r_lag))*self.omg_circ(self.r_lag))
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
        mllstart = m-self.r_lag-1
        nllstart = int(2**mllstart)
        if m==self.m_min: # first iteration
            n = int(2**m)
            self.y_cp = self.fast_transform(self.y_cp,0,m,m)
            self.kappanumap = arange(1,n+1,dtype=int)
            self.update_kappanumap(m-1,0,m)
            if self.ncv>0:
                self.yg_cp = self.fast_transform(self.yg_cp,0,m,m)
                self.beta_update(mllstart)
                self.kappanumap = arange(1,n+1,dtype=int)
                self.update_kappanumap(m-1,0,m)
        else: # any iteration after the first
            mnext = int(m-1)
            n = int(2**mnext)
            if not self.update_beta: # do not update the beta coefficients
                self.y_cp[-n:] = self.fast_transform(self.y_cp[-n:],0,mnext,mnext)
                self.y_cp = self.fast_transform(self.y_cp,mnext,mnext+1,mnext)
                self.kappanumap = hstack((self.kappanumap,int(2**(m-1))+self.kappanumap))
                self.update_kappanumap(m-1,mllstart,m) 
            else: # update beta
                self.y_cp = self.fast_transform(self.y_cp,0,m,m)
                self.yg_cp = self.fast_transform(self.yg_cp,0,m,m)
                self.beta_update(mllstart)
                self.kappanumap = hstack((self.kappanumap,2**(m-1)+self.kappanumap)).astype(int)
                self.update_kappanumap(m-1,mllstart,m)
        self.muhat = self.y_val.mean()+self.beta@self.cv_mu if self.ncv>0 else self.y_val.mean()
        stilde = sum(abs(self.y_cp[self.kappanumap[nllstart:2*nllstart]-1]))
        self.bounds = self.muhat+array([-1,1])*self.fudge(m)*stilde
        if self.check_cone:
            for l in range(self.l_star,m+1): # Storing the information for the necessary conditions
                c_tmp = self.omg_hat(m-l)*self.omg_circ(m-l)
                c_low = 1./(1+c_tmp)
                c_up = 1./(1-c_tmp)
                const1 = sum(abs(self.y_cp[self.kappanumap[int(2**(l-1)):int(2**l)]-1]))
                idx = int(l-self.l_star)
                self.c_stilde_low[idx] = max(self.c_stilde_low[idx],c_low*const1)
                if c_tmp < 1:
                    self.c_stilde_up[idx] = min(self.c_stilde_up[idx],c_up*const1)
            cone_violation = (self.c_stilde_low > self.c_stilde_up).any()
        else:
            cone_violation = False
        return self.muhat,self.bounds,cone_violation 

    def fast_transform(self, y2tf, mfrom, mto, m):
        for l in range(int(mfrom),int(mto)):
            nl = 2**l
            nmminlm1 = int(ceil(2**(m-l-1)))
            ptind_nl = hstack((tile(True,nl),tile(False,nl)))
            ptind = tile(ptind_nl,nmminlm1)
            coef = self.coefv(nl)
            coefv = tile(coef,nmminlm1)
            evenval = y2tf[ptind]
            oddval = y2tf[~ptind]
            y2tf[ptind] = (evenval+coefv*oddval)/2.
            y2tf[~ptind] = (evenval-coefv*oddval)/2.
        return y2tf

    def update_kappanumap(self, mfrom, mto, m):
        for l in range(int(mfrom),int(mto),-1):
            nl = 2**l
            oldone = abs(self.y_cp[self.kappanumap[1:int(nl)]-1]) # earlier values of kappa, don't touch first one
            newone = abs(self.y_cp[self.kappanumap[nl+1:2*nl]-1]) # later values of kappa,
            flip = where(newone>oldone)[0]+1 # which in the pair are the larger ones. change to matlab indexing
            if flip.size!=0:
                additive = arange(0,2**m-1,2**(l+1)).reshape((1,-1))
                flipall = (flip.reshape((-1,1))+additive)
                flipall = flipall.flatten('F').astype(int) # flatten column wise
                temp = self.kappanumap[nl+flipall] # then flip
                self.kappanumap[nl+flipall] = self.kappanumap[flipall] # them
                self.kappanumap[flipall] = temp # around
    
    def beta_update(self,mstart):
        kappa_approx = self.kappanumap[int(2**mstart):]-1 # kappa index used for fitting
        x4beta = self.yg_cp[kappa_approx]
        y4beta = self.y_cp[kappa_approx]
        self.beta = linalg.lstsq(x4beta,y4beta,rcond=None)[0]
        self.y_val = self.y_val-self.yg_val@self.beta # get new function values
        self.y_cp = self.y_cp-self.yg_cp@self.beta # redefine function
