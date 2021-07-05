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

    def __init__(self, stopping_crit, integrand, true_measure, discrete_distrib, coefv, 
        m_min, m_max, fudge, check_cone, ptransform, cast_complex,
        control_variates, control_variate_means, update_beta):
        """
        Args:
            stopping_crit (StoppingCriterion): a StoppingCriterion instance
            integrand (Integrand): an Integrand instance
            true_measure (TrueMeasure): A TrueMeasure instance
            discrete_distrib (DiscreteDistribution): a DiscreteDistribution instance
            coefv (method): function to return the coefficients of the transform based on nl
            m_min (int): initial n == 2^m_min
            m_max (int): max n == 2^m_max
            fudge (function): positive function multiplying the finite 
                sum of basis coefficients specified in the cone of functions
            check_cone (boolean): check if the function falls in the cone
            ptransform (str): periodization transform
            cast_complex (bool): need to cast as complex for fast trasform? 
            control_variates (list): list of integrand objects to be used as control variates. 
                Control variates are currently only compatible with single level problems. 
                The same discrete distribution instance must be used for the integrand and each of the control variates. 
            control_variate_means (list): list of means for each control variate
            update_beta (bool): update control variate beta coefficients at each iteration? 
        """
        self.parameters = ['solution','error_bound','n_total']
        self.stopping_crit = stopping_crit
        self.integrand = integrand
        self.true_measure = true_measure
        self.discrete_distrib = discrete_distrib
        # setup control variates
        self.cv = control_variates
        self.cv_mu = control_variate_means
        if isinstance(self.cv,Integrand):
            self.cv = [self.cv] # take a single integrand and make it into a list of length 1
        if isscalar(self.cv_mu):
            self.cv_mu = [self.cv_mu]
        if len(self.cv)!=len(self.cv_mu):
            raise ParameterError("list of control variates and list of control variate means must be the same.")
        for cv in self.cv:
            if cv.discrete_distrib != self.discrete_distrib:
                raise ParameterError('''
                        Each control variate's discrete distribution 
                        must be the same instance as the one for the main integrand.''')
        self.cv_mu = array(self.cv_mu) # column vector
        self.ncv = int(len(self.cv))
        self.update_beta = update_beta
        # Set Attributes
        self.coefv = coefv # fast transform 
        self.m_min = m_min
        self.m_max = m_max
        self.m = self.m_min
        self.n_total = 0  # total number of samples generated
        self.solution = nan
        self.r_lag = 4 # distance between coefficients summed and those computed
        self.l_star = self.m_min - self.r_lag # minimum gathering of points for the sums of DFT
        self.fudge = fudge
        self.omg_circ = lambda m: 2**(-m)
        self.omg_hat = lambda m: self.fudge(m)/((1+self.fudge(self.r_lag))*self.omg_circ(self.r_lag))
        # Initialize various sums of DFT terms for necessary conditions
        self.stilde = 0
        self.c_stilde_low = tile(-inf,int(self.m_max-self.l_star+1))
        self.c_stilde_up = tile(inf,int(self.m_max-self.l_star+1))
        self.check_cone = check_cone
        self.ptransform = ptransform
        self.cast_complex = cast_complex
        super(LDTransformData,self).__init__()

    def update_data(self):
        """ See abstract method. """
        # Generate sample values
        self.x = self.discrete_distrib.gen_samples(n_min=self.n_total,n_max=2**self.m)
        # update kappanumap and sub-computations
        if self.m==self.m_min:
            self._sub_update_data_first()
        else:
            self._sub_update_data_next()
        ## Compute Stilde
        nllstart = int64(2**(self.m-self.r_lag-1))
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
        
    def _sub_update_data_first(self): # first iteration
        # evaluate function (including CVs)
        n = int(2**self.m_min)
        if self.ncv==0: # not using control variates
            y = self.integrand.f_periodized(self.x,self.ptransform).squeeze()
            yval = y.copy()
        else: # using control variates
            ycv = zeros((n,1+self.ncv),dtype=float)
            ycv[:,0] = self.integrand.f_periodized(self.x,self.ptransform).squeeze()
            for i in range(self.ncv):
                ycv[:,i+1] = self.cv[i].f(self.x).squeeze()
            y = ycv[:,0].copy()
            yval = y.copy()
            yg = ycv[:,1:].copy()
            if self.cast_complex: yg = yg.astype(complex)
        if self.cast_complex: y = y.astype(complex)
        # fast transform
        for l in range(int(self.m_min)):
            nl = 2**l
            nmminlm1 = 2**(self.m_min-l-1)
            ptind_nl = hstack(( tile(True,nl), tile(False,nl) ))
            ptind = tile(ptind_nl,int(nmminlm1))
            coef = self.coefv(nl)
            coefv = tile(coef,int(nmminlm1))
            evenval = y[ptind]
            oddval = y[~ptind]
            y[ptind] = (evenval + coefv*oddval) / 2.
            y[~ptind] = (evenval - coefv*oddval) / 2.
            if self.ncv>0:
                evenval = yg[ptind]
                oddval = yg[~ptind]
                yg[ptind] = (evenval + coefv[:,None]*oddval) / 2.
                yg[~ptind] = (evenval - coefv[:,None]*oddval) / 2.
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
        if self.ncv==0: # not using control variates
            ynext = self.integrand.f_periodized(self.x,self.ptransform).squeeze()
            yval = hstack((self.yval,ynext))
        else: # using control variates
            ycvnext = zeros((n,1+self.ncv),dtype=float)
            ycvnext[:,0] = self.integrand.f_periodized(self.x,self.ptransform).squeeze()
            for i in range(self.ncv):
                ycvnext[:,i+1] = self.cv[i].f(self.x).squeeze()
            ynext = ycvnext[:,0] - ycvnext[:,1:]@self.beta
            yval = hstack((self.yval,ynext))
        if self.cast_complex: 
            ynext = ynext.astype(complex)
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
      