""" Definition for MeanVarDataRep, a concrete implementation of AccumulateData """

from ._accumulate_data import AccumulateData
from ..util import CubatureWarning
from numpy import array, nan, zeros, tile, inf, hstack, exp, pi, arange, where, complex128
import warnings


class CubatureData(AccumulateData):
    """
    Accumulated data relavent to cubature algorithms
    """

    parameters = ['n_total','solution','r_lag']

    def __init__(self, stopping_criterion, integrand, m_min, m_max, fudge):
        """
        Initialize data instance

        Args:
            stopping_criterion (StoppingCriterion): a StoppingCriterion instance
            integrand (Integrand): an Integrand instance
            m_min (int): initial n == 2^m_min
            m_max (int): max n == 2^m_max
            fudge (function): positive function multiplying the finite
                              sum of Fast Fourier coefficients specified 
                              in the cone of functions
        """
        # Extract attributes from integrand
        self.stopping_criterion = stopping_criterion
        self.integrand = integrand
        self.measure = self.integrand.measure
        self.distribution = self.measure.distribution
        # Set Attributes
        self.m_min = m_min
        self.m_max = m_max
        self.m = self.m_min
        self.n_total = 0  # total number of samples generated
        self.solution = nan
        self.r_lag = 4 # distance between coefficients summed and those computed
        self.l_star = self.m_min - self.r_lag # minimum gathering of points for the sums of DFT
        self.yval = array([]) # hold y values
        self.y = array([],dtype=complex128) # hold transformed y values
        self.kappanumap = arange(1,2**self.m+1,dtype=int)
        self.fudge = fudge
        self.omg_circ = lambda m: 2**(-m)
        self.omg_hat = lambda m: self.fudge(m)/((1+self.fudge(self.r_lag))*self.omg_circ(self.r_lag))
        # Initialize various sums of DFT terms for necessary conditions
        self.stilde = 0
        self.c_stilde_low = tile(-inf,int(self.m_max-self.l_star+1))
        self.c_stilde_up = tile(inf,int(self.m_max-self.l_star+1))
        super().__init__()

    def update_data(self):
        """ Update data """
        # Generate sample values
        x_lat = self.measure.gen_samples(n_min=self.n_total,n_max=2**self.m)
        ynext = self.integrand.f(x_lat).squeeze()
        self.yval = hstack((self.yval,ynext))
        ynext = ynext.astype(complex)
        mnext = self.m-1 if self.m>self.m_min else self.m_min
        ## Compute initial FFT on next points
        for l in range(int(mnext)):
            nl = 2**l
            nmminlm1 = 2**(mnext-l-1)
            ptind_nl = hstack((tile(True,nl),tile(False,nl)))
            ptind = tile(ptind_nl,int(nmminlm1))
            coef = exp(-2*pi*1j*arange(nl)/(2*nl))
            coefv = tile(coef,int(nmminlm1))
            evenval = ynext[ptind]
            oddval = ynext[~ptind]
            ynext[ptind] = (evenval+coefv*oddval)/2
            ynext[~ptind] = (evenval-coefv*oddval)/2
        self.y = hstack((self.y,ynext))
        if self.m > self.m_min: # already generated samples
            ## Compute FFT on all points
            nl = 2**mnext
            ptind = hstack((tile(True,int(nl)),tile(False,int(nl))))
            coefv = exp(-2*pi*1j*arange(nl)/(2*nl))
            evenval = self.y[ptind]
            oddval = self.y[~ptind]
            self.y[ptind] = (evenval+coefv*oddval)/2
            self.y[~ptind] = (evenval-coefv*oddval)/2
            # combine self.kappanumap from previous
            self.kappanumap = hstack((self.kappanumap, 2**(self.m-1)+self.kappanumap)).astype(int) #initialize map
            ls = arange(int(self.m-1),int(self.m-self.r_lag-1),-1, dtype=int)
        else: # self.m == self.m_min (first computation)
            ls = arange(self.m-1,0,-1, dtype=int)
        ## Update self.kappanumap
        for l in ls:
            nl = 2**l
            oldone = abs(self.y[self.kappanumap[1:int(nl)]-1]) # earlier values of kappa, don't touch first one
            newone = abs(self.y[self.kappanumap[nl+1:2*nl]-1]) # later values of kappa,
            flip = where(newone>oldone)[0]+1 # which in the pair are the larger ones. change to matlab indexing
            if flip.size != 0:
                additive = arange(0,2**self.m-1,2**(l+1)).reshape((1,-1))
                flipall = (flip.reshape((-1,1)) + additive)
                flipall = flipall.flatten('F').astype(int) # flatten column wise
                temp = self.kappanumap[nl+flipall] # then flip
                self.kappanumap[nl+flipall] = self.kappanumap[flipall] # them
                self.kappanumap[flipall] = temp # around   
        ## Compute Stilde
        nllstart = int(2**(self.m-self.r_lag-1))
        self.stilde = sum(abs(self.y[self.kappanumap[nllstart:2*nllstart]-1]))
        # Necessary conditions
        for l in range(int(self.l_star),int(self.m+1)): # Storing the information for the necessary conditions
            c_tmp = self.omg_hat(self.m-l)*self.omg_circ(self.m-l)
            c_low = 1/(1+c_tmp)
            c_up = 1/(1-c_tmp)
            const1 = sum(abs(self.y[self.kappanumap[int(2**(l-1)):int(2**l)]-1]))
            idx = int(l-self.l_star)
            self.c_stilde_low[idx] = max(self.c_stilde_low[idx],c_low*const1)
            if c_tmp < 1:
                self.c_stilde_up[idx] = min(self.c_stilde_up[idx],c_up*const1)
        if (self.c_stilde_low > self.c_stilde_up).any():
            warnings.warn('An element of c_stilde_low > c_stilde_up', CubatureWarning)
        ## Approximate integral
        self.solution = self.yval.mean()
        # update total samples
        self.n_total = 2**self.m # updated the total evaluations
