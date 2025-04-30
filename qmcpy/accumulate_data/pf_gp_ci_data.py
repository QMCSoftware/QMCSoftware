from ._accumulate_data import AccumulateData
from ..util import ExactGPyTorchRegressionModel
import numpy as np
from scipy.stats import norm
import warnings
import gpytorch 
import torch

def _get_phi(gp, x):
    yhat,yhatstd = gp.predict(x)
    with np.errstateall='ignore'): z = yhat/yhatstd
    return norm.cdf(z)
def _error_udens_from_phi(phi):
    return 2*min(1-phi,phi)
def _error_udens(gp, x):
    return _error_udens_from_phi(_get_phi(gp,x))

class PFGPCIData(AccumulateData):
    """ Update and store data for the PFGPCI StoppingCriterion. """

    def __init__(self, stopping_crit, integrand, true_measure, discrete_distrib, dnb2, n_approx, alpha, refit, 
        gpytorch_prior_mean, gpytorch_prior_cov, gpytorch_likelihood, gpytorch_marginal_log_likelihood_func,
        torch_optimizer_func,gpytorch_train_iter,gpytorch_use_gpu,verbose,approx_true_solution):
        self.parameters = ['solution','error_bound','bound_low','bound_high','n_total','time_integrate']
        self.stopping_crit = stopping_crit
        self.integrand = integrand
        self.true_measure = true_measure
        self.discrete_distrib = discrete_distrib
        self.d = self.discrete_distrib.d
        self.dnb2 = dnb2
        self.alpha = alpha
        self.refit = refit
        self.qmc_pts = self.dnb2.gen_samples(n_approx)
        self.gpytorch_prior_mean = gpytorch_prior_mean
        self.gpytorch_prior_cov = gpytorch_prior_cov
        self.gpytorch_likelihood = gpytorch_likelihood
        self.gpytorch_marginal_log_likelihood_func = gpytorch_marginal_log_likelihood_func
        self.torch_optimizer_func = torch_optimizer_func
        self.gpytorch_train_iter = gpytorch_train_iter
        self.gpytorch_use_gpu = gpytorch_use_gpu
        self.verbose = verbose
        self.approx_true_solution = approx_true_solution
        self.n_batch = []
        self.emr = []
        self.error_bounds = []
        self.ci_low = []
        self.ci_high = []
        self.solutions = []
        self.x = np.empty((0,self.d),dtype=float)
        self.y = np.array([],dtype=float)
        self.gpyt_model = ExactGPyTorchRegressionModel(
            x_t = self.x, y_t = self.y,
            prior_mean = self.gpytorch_prior_mean,
            prior_cov = self.gpytorch_prior_cov,
            likelihood = self.gpytorch_likelihood,
            use_gpu = self.gpytorch_use_gpu)   
        self.saved_gps = [self.gpyt_model.state_dict()]
        super(PFGPCIData,self).__init__()

    def update_data(self, batch_count, xdraw, ydrawtf):
        self.n_batch.append(len(xdraw))
        self.x,self.y = np.vstack([self.x,xdraw]),np.hstack([self.y,ydrawtf])
        if batch_count==0 or self.refit:
            self.gpyt_model = ExactGPyTorchRegressionModel(
                x_t = self.x, y_t = self.y,
                prior_mean = self.gpytorch_prior_mean,
                prior_cov = self.gpytorch_prior_cov,
                likelihood = self.gpytorch_likelihood,
                use_gpu = self.gpytorch_use_gpu)
            self.gpyt_model.fit(
                optimizer = self.torch_optimizer_func(self.gpyt_model),
                mll = self.gpytorch_marginal_log_likelihood_func(self.gpyt_model.likelihood,self.gpyt_model),
                training_iter = self.gpytorch_train_iter,
                verbose = self.verbose)
        else:
            self.gpyt_model = self.gpyt_model.add_data(xdraw,ydrawtf)
            torch.cuda.empty_cache()          
        self.saved_gps.append(self.gpyt_model.state_dict())
        phi = _get_phi(self.gpyt_model,self.qmc_pts)
        self.solutions.append(np.mean(phi>=.5))
        self.emr.append(np.mean(minimum(phi,1-phi)))
        gamma = self.emr[-1]/self.alpha
        self.ci_low.append(maximum(self.solutions[-1]-gamma,0))
        self.ci_high.append(minimum(self.solutions[-1]+gamma,1))
        self.error_bounds.append(maximum(self.solutions[-1]-self.ci_low[-1],self.ci_high[-1]-self.solutions[-1]))

    def get_results_dict(self):
        df = {'n_sum':self.n_sum, 'n_batch':self.n_batch, 'error_bounds':self.error_bounds, 'ci_low':self.ci_low, 'ci_high':np.array(self.ci_high), 'solutions':np.array(self.solutions)}
        if self.approx_true_solution: df['solutions_ref'],df['error_ref'],df['in_ci'] = self.solutions_ref,self.error_ref,self.in_ci
        return df 
    
    def plot(self, trace_only=False, **kwargs):
        from matplotlib import pyplot
        if self.d==1 and not trace_only and self.saved_gps!=[]: fig,gs = self.plot_1d(**kwargs)
        elif self.d==2 and not trace_only and self.saved_gps!=[]: fig,gs = self.plot_2d(**kwargs)
        else:
            from matplotlib import pyplot,gridspec
            fig = pyplot.figure(constrained_layout=False,figsize=(8,4))
            gs = gridspec.GridSpec(1,2,figure=fig)
        axtrace = fig.add_subplot(gs[-1,:])
        if self.approx_true_solution: axtrace.plot(.5+np.arange(len(self.n_batch)),self.solutions_ref,color='c',label=r'$P(g)$')
        axtrace.plot(.5+np.arange(len(self.n_batch)),self.solutions,'-o',color='k',label=r'$\hat{P}_n^\mathrm{QMC}$')
        axtrace.fill_between(.5+np.arange(len(self.n_batch)),self.ci_low,self.ci_high,color='k',alpha=.15)
        axtrace.plot(.5+np.arange(len(self.n_batch)),self.error_bounds,'-o',color='m',label=r'$\hat{\gamma}_n^{\text{QMC}}$')
        if self.approx_true_solution: axtrace.plot(.5+np.arange(len(self.n_batch)),self.error_ref,'-o',color='b',label=r'$\vert \hat{P}_n^{\text{QMC}} - P(g) \vert$')
        axtrace.set_xlim([0,len(self.n_batch)])
        axtrace.set_xticks([])
        axtrace.set_yscale('log',base=10)
        #axtrace.set_ylim([0,1]); axtrace.set_yticks([0,1])
        for spine in ['top','right','bottom']: axtrace.spines[spine].set_visible(False)
        axtrace.legend(loc='lower left',bbox_to_anchor=(0.05, -.2, .9, .102),mode='expand',ncol=4,prop={'size':18},frameon=False)
        return fig
    def plot_1d(self, meshticks=1025, ci_percentage=.95, **kwargs):
        from matplotlib import pyplot,gridspec
        beta = norm.ppf(mean([ci_percentage,1]))
        n_batches = len(self.n_batch)
        fig = pyplot.figure(constrained_layout=False,figsize=(4*n_batches,4*3))
        gs = gridspec.GridSpec(3,n_batches,figure=fig)
        xticks = np.linspace(0,1,meshticks)[1:-1]
        if self.approx_true_solution: 
            yticks = self.integrand.f(xticks[:,None]).squeeze()
            ytickstf = self.stopping_crit._affine_tf(yticks)
        for j in range(n_batches):
            i0,i1 = 0 if j==0 else self.n_sum[j-1],self.n_sum[j]
            gpyt_model = ExactGPyTorchRegressionModel(
                x_t = self.x[:i0], y_t = self.y[:i0],
                prior_mean = self.gpytorch_prior_mean,
                prior_cov = self.gpytorch_prior_cov,
                likelihood = self.gpytorch_likelihood,
                use_gpu = self.gpytorch_use_gpu)
            gpyt_model.load_state_dict(self.saved_gps[j])
            ax0j = fig.add_subplot(gs[0,j])
            mr_udens = _error_udens(gpyt_model,xticks[:,None])
            ax0j.fill_between(xticks,mr_udens,color='k',alpha=.5)
            ax0j.scatter(self.x[i0:i1].squeeze(),np.zeros(i1-i0)+.015,color='r',s=25)
            ax0j.set_ylim([0,1])
            ax0j.set_yticks([0,1])
            gpyt_model = ExactGPyTorchRegressionModel(
                x_t = self.x[:i1], y_t = self.y[:i1],
                prior_mean = self.gpytorch_prior_mean,
                prior_cov = self.gpytorch_prior_cov,
                likelihood = self.gpytorch_likelihood,
                use_gpu = self.gpytorch_use_gpu)
            gpyt_model.load_state_dict(self.saved_gps[j+1])
            ax1j = fig.add_subplot(gs[1,j],sharey=None if j==0 else ax1j)
            if self.approx_true_solution: ax1j.plot(xticks,ytickstf,color='c',label=r'f(x)',linewidth=2)
            gp_mean,gp_std = gpyt_model.predict(xticks[:,None])
            ax1j.plot(xticks,gp_mean,color='k',label='gp')
            ax1j.fill_between(xticks,gp_mean-beta*gp_std,gp_mean+beta*gp_std,color='k',alpha=.25)
            ax1j.axhline(y=0,color='lightgreen')
            ax1j.scatter(self.x[:i0].squeeze(),self.y[:i0],color='b',label='old pts')
            ax1j.scatter(self.x[i0:i1].squeeze(),self.y[i0:i1],color='r',label='new pts')
            ax1j.set_xlabel(r'$u$')
            if j==0:
                #ax1j.set_ylabel(r'$y$')
                ax0j.set_ylabel(r'$2\mathrm{ERR}_n(u)$')
        for ax in fig.axes:
            ax.set_xlim([0,1])
            ax.set_xticks([0,1])
        return fig,gs
    def plot_2d(self, meshticks=257, clevels=32, **kwargs):
        from matplotlib import pyplot,gridspec,cm
        n_batches = len(self.n_batch)
        fig = pyplot.figure(constrained_layout=False,figsize=(4*n_batches,4*5))
        gs = gridspec.GridSpec(5 if self.approx_true_solution else 4,n_batches,figure=fig)
        x0mesh,x1mesh = meshgrid(np.linspace(0,1,meshticks)[1:-1],np.linspace(0,1,meshticks)[1:-1])
        xquery = np.vstack([x0mesh.flatten(),x1mesh.flatten()]).T
        if self.approx_true_solution:
            yquery = self.integrand.f(xquery).squeeze()
            ymeshtf = self.stopping_crit._affine_tf(yquery).reshape(x0mesh.shape)
        for j in range(n_batches):
            row_idx = 0
            i0,i1 = 0 if j==0 else self.n_sum[j-1],self.n_sum[j]
            if self.approx_true_solution:
                ax0j = fig.add_subplot(gs[row_idx,j])
                ax0j.contourf(x0mesh,x1mesh,ymeshtf,cmap=cm.Greys,vmin=ymeshtf.min(),vmax=ymeshtf.max(),levels=clevels)
                ax0j.contour(x0mesh,x1mesh,ymeshtf,colors='lightgreen',levels=[0],linewidths=5)
                ax0j.set_title(r'$g(\boldsymbol{u})$')
                row_idx += 1
            gpyt_model = ExactGPyTorchRegressionModel(
                x_t = self.x[:i0], y_t = self.y[:i0],
                prior_mean = self.gpytorch_prior_mean,
                prior_cov = self.gpytorch_prior_cov,
                likelihood = self.gpytorch_likelihood,
                use_gpu = self.gpytorch_use_gpu)
            gpyt_model.load_state_dict(self.saved_gps[j])
            ax1j = fig.add_subplot(gs[row_idx,j])
            udens_mr = _error_udens(gpyt_model,xquery).reshape(x0mesh.shape)
            ax1j.contourf(x0mesh,x1mesh,udens_mr,cmap=cm.Greys,levels=clevels,vmin=0,vmax=1)
            ax1j.scatter(self.x[i0:i1,0],self.x[i0:i1,1],color='r')
            ax1j.set_title(r'$2\mathrm{ERR}_n(\boldsymbol{u})$')
            row_idx += 1
            gpyt_model = ExactGPyTorchRegressionModel(
                x_t = self.x[:i0], y_t = self.y[:i0],
                prior_mean = self.gpytorch_prior_mean,
                prior_cov = self.gpytorch_prior_cov,
                likelihood = self.gpytorch_likelihood,
                use_gpu = self.gpytorch_use_gpu)
            gpyt_model.load_state_dict(self.saved_gps[j+1])
            gp_mean,gp_std = gpyt_model.predict(xquery)
            gp_mean_mesh,gp_std_mesh = gp_mean.reshape(x0mesh.shape),gp_std.reshape(x0mesh.shape)
            ax2j = fig.add_subplot(gs[row_idx,j])
            ax2j.contourf(x0mesh,x1mesh,gp_mean_mesh,cmap=cm.Greys,vmin=gp_mean.min(),vmax=gp_mean.max(),levels=clevels)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ax2j.contour(x0mesh,x1mesh,gp_mean_mesh,colors='lightgreen',levels=[0],linewidths=5)
            ax2j.set_title(r'$m_n(\boldsymbol{u})$')
            row_idx += 1
            ax3j = fig.add_subplot(gs[row_idx,j])            
            ax3j.contourf(x0mesh,x1mesh,gp_std_mesh,cmap=cm.Greys,levels=clevels)
            ax3j.set_title(r'$\sigma_n(\boldsymbol{u})$')
            for ax in [ax2j,ax3j]:
                ax.scatter(self.x[:i0,0],self.x[:i0,1],color='b')
                ax.scatter(self.x[i0:i1,0],self.x[i0:i1,1],color='r')
        for ax in fig.axes:
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            ax.set_xlabel(r'$u_1$')
            ax.set_aspect(1)
            ax.set_xticks([0,1])
            ax.set_yticks([0,1])
            ax.set_ylabel(r'$u_2$')
        return fig,gs
