from ._accumulate_data import AccumulateData
from ..util import CubatureWarning
from numpy import array, nan, zeros, tile, inf, hstack, arange, where
import warnings


class LDTransformData(AccumulateData):

    parameters = ['n_total','solution','r_lag']

    def __init__(self, stopping_criterion, integrand, basis_transform, m_min, m_max, fudge, check_cone):
        """
        Args:
            stopping_criterion (StoppingCriterion): a StoppingCriterion instance
            integrand (Integrand): an Integrand instance
            basis_transform (method): Transform ynext, combine with y, and then transform all points. 
                For cub_lattice this is Fast Fourier Transform (FFT). 
                For cub_sobol this is Fast Walsh Transform (FWT)
            m_min (int): initial n == 2^m_min
            m_max (int): max n == 2^m_max
            fudge (function): positive function multiplying the finite 
                sum of basis coefficients specified in the cone of functions
            check_cone (boolean): check if the function falls in the cone
        """
        # Extract attributes from integrand
        self.stopping_criterion = stopping_criterion
        self.integrand = integrand
        self.measure = self.integrand.measure
        self.distribution = self.measure.distribution
        # Set Attributes
        self.ft = basis_transform # fast transform 
        self.m_min = m_min
        self.m_max = m_max
        self.m = self.m_min
        self.n_total = 0  # total number of samples generated
        self.solution = nan
        self.r_lag = 4 # distance between coefficients summed and those computed
        self.l_star = self.m_min - self.r_lag # minimum gathering of points for the sums of DFT
        self.yval = array([]) # hold y values
        self.y = array([]) # hold transformed y values
        self.kappanumap = arange(1,2**self.m+1,dtype=int)
        self.fudge = fudge
        self.omg_circ = lambda m: 2**(-m)
        self.omg_hat = lambda m: self.fudge(m)/((1+self.fudge(self.r_lag))*self.omg_circ(self.r_lag))
        # Initialize various sums of DFT terms for necessary conditions
        self.stilde = 0
        self.c_stilde_low = tile(-inf,int(self.m_max-self.l_star+1))
        self.c_stilde_up = tile(inf,int(self.m_max-self.l_star+1))
        self.check_cone = check_cone
        super().__init__()

    def update_data(self):
        """ See abstract method. """
        # Generate sample values
        x = self.distribution.gen_samples(n_min=self.n_total,n_max=2**self.m)
        ynext = self.integrand.f(x).squeeze()
        self.yval = hstack((self.yval,ynext))
        # Compute fast basis transform
        self.y = self.ft(self.y, ynext)
        ## Update self.kappanumap
        if self.y.size == ynext.size:
            ls = arange(self.m-1,0,-1, dtype=int)
        else:
            ls = arange(int(self.m-1),int(self.m-self.r_lag-1),-1, dtype=int)
            # combine self.kappanumap from previous
            self.kappanumap = hstack((self.kappanumap, 2**(self.m-1)+self.kappanumap)).astype(int) #initialize map
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
        ## Approximate integral
        self.solution = self.yval.mean()
        # update total samples
        self.n_total = 2**self.m # updated the total evaluations
        # Necessary conditions
        if not self.check_cone: return # don't check if the function falls in the cone
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
        
