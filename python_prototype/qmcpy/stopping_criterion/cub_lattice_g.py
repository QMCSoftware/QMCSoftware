""" Definition for CubLattice_g, a concrete implementation of StoppingCriterion """

from ._stopping_criterion import StoppingCriterion
from ..accum_data import CubatureData
from ..util import NotYetImplemented, Parameter

from numpy import log2, inf, zeros, ones, tile, \
                    exp, pi, arange, vstack, where, maximum, minimum, \


class CubLattice_g(StoppingCriterion):
    """
    Stopping Criterion quasi-Monte Carlo method using rank-1 Lattices cubature over
    a d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Fourier coefficients cone decay assumptions.

    Guarantee
        This algorithm computes the integral of real valued functions in :math:`[0,1]^d`
        with a prescribed generalized error tolerance. The Fourier coefficients
        of the integrand are assumed to be absolutely convergent. If the
        algorithm terminates without warning messages, the output is given with
        guarantees under the assumption that the integrand lies inside a cone of
        functions. The guarantee is based on the decay rate of the Fourier
        coefficients. For integration over domains other than :math:`[0,1]^d`, this cone
        condition applies to :math:`f \circ \psi` (the composition of the
        functions) where :math:`\psi` is the transformation function for :math:`[0,1]^d` to
        the desired region. For more details on how the cone is defined, please
        refer to the references below.
    """

    def __init__(self, discrete_distrib, true_measure,
                 inflate=1.2, alpha=0.01,
                 abs_tol=1e-2, rel_tol=0,
                 n_init=2**10, n_max=2**35,
                 fudge = lambda m: 5*2**(-m)):
        """
        Args:
            discrete_distrib
            true_measure (DiscreteDistribution): an instance of DiscreteDistribution
            inflate (float): inflation factor when estimating variance
            alpha (float): significance level for confidence interval
            abs_tol (float): absolute error tolerance
            rel_tol (float): relative error tolerance
            n_max (int): maximum number of samples
            fudge (function): positive function multiplying the finite
                              sum of Fast Fourier coefficients specified 
                              in the cone of functions
        """
        # Input Checks
        levels = len(true_measure)
        if levels != 1:
            raise NotYetImplemented('''
                cub_lattice_g not implemented for multi-level problems.
                Use CLT stopping criterion with an iid distribution for multi-level problems ''')
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_max = n_max
        self.fudge = fudge
        self.stage = "sigma"
        # Construct Data Object to House Integration data
        m_min = log2(n_init)
        m_max = log2(n_max)
        if m_min%1 != 0 or m_max%1 != 0:
            warning_s = ' n_init and n_max must be a powers of 2. Using n_init = 2**5 and n_max=2**35'
            warnings.warn(warning_s, ParameterWarning)
            m_min = 5
            m_max = 35 
        self.data = CubatureData(len(true_measure), m_min, m_max)
        # Variables used by algorithm
        self.exit_len = 2
        self.r_lag = 4 # distance between coefficients summed and those computed
        self.l_star = self.data.mmin - r_lag # minimum gathering of points for the sums of DFT
        self.omg_circ = lambda m: 2**(-m)
        self.omg_hat = lambda m: self.fudge(m)/(1+self.fudge(r_lag))*omg_circ(r_lag)
        self.Stilde = zeros((self.mmax-self.mmin+1,1)) # initialize sum of DFT terms
        self.CStilde_low = -inf*ones((1,self.mmax-l_star+1)) # initialize various sums of DFT terms for necessary conditions
        self.CStilde_up = inf*ones((1,self.mmax-l_star+1)) # initialize various sums of DFT terms for necessary conditions
        self.errest = zeros((self.mmax-self.mmin+1,1)) # initialize error estimates
        self.appxinteg = zeros((self.mmax-self.mmin+1,1)) # initialize approximations to integral
        self.exit = zeros((1,exit_len)) # we start the algorithm with all warning flags down
        # Verify Compliant Construction
        allowed_distribs = ["Lattice"]
        super().__init__(discrete_distrib, allowed_distribs)

    def stop_yet(self):
        """ Determine when to stop """
        raise Exception("Not yet implemented")
        
        if self.stage == 'sigma':

        
        # generate Lattice samples into `y`
        
        yval = y
        ## Compute initial FFT so y contains the FFT coefficients          
        y = self.fft(y,ls=range(self.mmin))
        ## Approximate integral
        q = yval.mean()
        appxinteg[0] = q
        ## Create kappanumap implicitly from the data
        kappanumap = arange(1,self.data.n+1) # initialize map
        kappanumap = self.update_kappanumap(kappanumap,ls=range(self.mmin-1,0,-1),m_up=self.mmin)
        ## Compute Stilde
        nllstart = 2**(self.mmin-r_lag-1)
        Stilde[0] = sum(abs(y[kappanumap[nllstart:2*nllstart+1]]))
        self.bound_err = self.fudge(self.mmin)*Stilde[0]
        errest[0] = self.bound_err
        # Necessary conditions
        for l in range(l_star,self.mmin+1): # Storing the information for the necessary conditions
            C_low = 1/(1+omg_hat(self.mmin-l)*omg_circ(self.mmin-l))
            C_up = 1/(1-omg_hat(self.mmin-l)*omg_circ(self.mmin-l))
            CStilde_low[l-l_star] = maximum(CStilde_low[l-l_star],C_low*sum(abs(y[kappanumap[2**(l-1):2**l+1]])))
            if (omg_hat(self.mmin-l)*omg_circ(self.mmin-l) < 1):
                CStilde_up[l-l_star] = minimum(CStilde_up[l-l_star],C_up*sum(abs(y[kappanumap[2**(l-1):2**l+1]])))
        if (CStilde_low > CStilde_up).any():
            self.exit[2] = true
        # Check the end of the algorithm
        ub = max(self.abstol, self.reltol*abs(q + errest[0]))
        lb = max(self.abstol, self.reltol*abs(q - errest[0]))
        q = q - errest[0]*(ub-lb) / (ub+lb) # Optimal estimator
        appxinteg[0] = q

        
        is_done = False
        if 4*errest[0]**2/(ub+lb))**2 <= 1:
            is_done = True
        elif self.mmin == self.mmax: # We are on our max budget and did not meet the error condition => overbudget
            out_param.exit[1] = True
            is_done = True
        
        ## Loop over m
        for m in range(self.mmin+1,self.mmax+1):
            if is_done:
                break
            self.data.n = 2**m
            mnext = m-1
            nnext = 2**mnext
            
            # generate Lattice samples into `ynext`
            
            
            yval = vstack((yval,ynext))
            ## Compute initial FFT on next points
            ynext = self.fft(ynext, ls=range(0,mnext))
            y = vstack((y,ynext))
            ## Compute FFT on all points
            y = self.fft(y, ls=[mnext])
            ## Update kappanumap
            kappanumap = vstack((kappanumap, 2^(m-1)+kappanumap)) #initialize map
            self.update_kappanumap(kappanumap,ls=range(m-1,m-r_lag-1,-1),m_up=m)
            
## LEFT OFF HERE
            ## Compute Stilde
            nllstart=int64(2^(m-r_lag-1))
            meff=m-self.mmin+1
            Stilde(meff)=sum(abs(y(kappanumap(nllstart+1:2*nllstart))))
            out_param.bound_err=out_param.fudge(m)*Stilde(meff)
            errest(meff)=out_param.bound_err
            
            # Necessary conditions
            for l = l_star:m # Storing the information for the necessary conditions
                    C_low = 1/(1+omg_hat(m-l)*omg_circ(m-l))
                    C_up = 1/(1-omg_hat(m-l)*omg_circ(m-l))
                    CStilde_low(l-l_star+1) = max(CStilde_low(l-l_star+1),C_low*sum(abs(y(kappanumap(2^(l-1)+1:2^l)))))
                    if (omg_hat(m-l)*omg_circ(m-l) < 1)
                        CStilde_up(l-l_star+1) = min(CStilde_up(l-l_star+1),C_up*sum(abs(y(kappanumap(2^(l-1)+1:2^l)))))
                    end
            end
            
            if any(CStilde_low(:) > CStilde_up(:))
                out_param.exit(2) = true
            end
            
            ## Approximate integral
            q=mean(yval)
            
            # Check the end of the algorithm
            q = q - errest(meff)*(max(out_param.abstol, out_param.reltol*abs(q + errest(meff)))...
                    - max(out_param.abstol, out_param.reltol*abs(q - errest(meff))))/...
                    (max(out_param.abstol, out_param.reltol*abs(q + errest(meff)))...
                    + max(out_param.abstol, out_param.reltol*abs(q - errest(meff)))) # Optimal estimator
            appxinteg(meff)=q
            
            if 4*errest(meff)^2/(max(out_param.abstol, out_param.reltol*abs(q + errest(meff)))...
                    + max(out_param.abstol, out_param.reltol*abs(q - errest(meff))))^2 <= 1
                out_param.time=toc(t_start)
                is_done = true
            elseif m == self.mmax # We are on our max budget and did not meet the error condition => overbudget
                out_param.exit(1) = true
            end
        end
        
        # Decode the exit structure
        exit_str=2.^(0:exit_len-1).*out_param.exit
        exit_str(out_param.exit==0)=[]
        if numel(exit_str)==0
            out_param.exitflag=0
        else
            out_param.exitflag=exit_str
        end
        
        out_param = rmfield(out_param,'exit')


    def fft(self, y, ls):
        for l in ls:
            nl = 2**l
            nmminlm1 = 2**(self.mmin-l-1)
            ptind_nl = hstack((tile(True,nl),tile(False,nl)))
            ptind = tile(ptind_nl,nmminlm1)
            coef = exp(-2*pi*1j*arange(nl)/(2*nl)).reshape((nl,1))
            coefv = tile(coef,(nmminlm1,1))
            evenval = y[ptind]
            oddval = y[~ptind]
            y[ptind] = (evenval+coefv*oddval)/2
            y[~ptind] = (evenval-coefv*oddval)/2
        return y

    def update_kappanumap(self, kappanumap, ls, m_up):
        for l in ls:
            nl = 2**l
            oldone = abs(y[kappanumap[1:nl]]) # earlier values of kappa, don't touch first one
            newone=abs(y[kappanumap[nl+1:2*nl]]) # later values of kappa,
            flip = where(newone>oldone)[0] # which in the pair are the larger ones
            if flip.size != 0:
                flipall = flip + arange(0,2**m_up-1+2**(l+1),2**(l+1))
                temp = kappanumap[nl+flipall] # then flip
                kappanumap[nl+flipall] = kappanumap[flipall] # them
                kappanumap[flipall] = temp # around
        return kappanumap
        
