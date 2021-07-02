from ._accumulate_data import AccumulateData
from ..integrand._integrand import Integrand
from ..util import ParameterError
from time import time
from numpy import *

class MeanVarData(AccumulateData):
    """ Update and store mean and variance estimates. """

    parameters = ['levels','solution','n','n_total','error_bound','confid_int']
    EPS = finfo(float32).eps

    def __init__(self, stopping_crit, integrand, true_measure, discrete_distrib, n_init, control_variates, control_variate_means):
        """
        Args:
            stopping_crit (StoppingCriterion): a StoppingCriterion instance
            integrand (Integrand): an Integrand instance
            true_measure (TrueMeasure): A TrueMeasure instance
            discrete_distrib (DiscreteDistribution): a DiscreteDistribution instance  
            n_init (int): initial number of samples
            control_variates (list): list of integrand objects to be used as control variates. 
                Control variates are currently only compatible with single level problems. 
                The same discrete distribution instance must be used for the integrand and each of the control variates. 
            control_variate_means (list): list of means for each control variate
        """
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
                        must be the same instance as the one for te main integrand.''')
        self.cv_mu = array(self.cv_mu).reshape((-1,1)) # column vector
        self.ncv = int(len(self.cv))
        # Set Attributes
        if self.integrand.leveltype=='fixed-multi':
            self.levels = len(self.integrand.dimensions)
            if not self.cv == []:
                raise ParameterError("Control variates are currently only supported for single-level problems.")
        else:
            self.levels = 1
        self.solution = nan
        self.muhat = full(self.levels, inf)  # sample mean
        self.sighat = full(self.levels, inf)  # sample standard deviation
        self.t_eval = zeros(self.levels)  # processing time for each integrand
        self.n = tile(n_init, self.levels) # currnet number of samples
        self.n_total = 0  # total number of samples
        self.confid_int = array([-inf, inf])  # confidence interval for solution
        super(MeanVarData,self).__init__()

    def update_data(self):
        """ See abstract method. """
        for l in range(self.levels):
            t_start = time() # time the integrand values
            if self.integrand.leveltype=='fixed-multi':
                # reset dimension
                new_dim = self.integrand._dim_at_level(l)
                self.true_measure._set_dimension_r(new_dim)
                samples = self.discrete_distrib.gen_samples(n=self.n[l])
                y = self.integrand.f(samples,l=l).squeeze()
            else:
                n = int(self.n[l])
                samples = self.discrete_distrib.gen_samples(n)
                y = self.integrand.f(samples).squeeze()
                if self.ncv>0:
                    # using control variates
                    cvdata = zeros((n,self.ncv),dtype=float)
                    for i in range(self.ncv):
                        cvdata[:,i] = self.cv[i].f(samples).squeeze()
                    cvmuhats = cvdata.mean(0)
                    if not hasattr(self,'beta_hat'):
                        # approximate control varite coefficient
                        x4beta = cvdata-cvmuhats[None,:]
                        y4beta = y-y.mean()
                        self.beta_hat = linalg.lstsq(x4beta,y4beta,rcond=None)[0].reshape((-1,1))
                    cvterm = self.beta_hat.T@(cvdata-self.cv_mu.T).T
                    y = y-cvterm.squeeze() # use control variates and apprixmated coefficient to 
            self.t_eval[l] = max( (time()-t_start)/self.n[l], self.EPS) 
            self.sighat[l] = y.std() # compute the sample standard deviation
            self.muhat[l] = y.mean() # compute the sample mean
            self.n_total += self.n[l] # add to total samples
        self.solution = self.muhat.sum() # tentative solution
