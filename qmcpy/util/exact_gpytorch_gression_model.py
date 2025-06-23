import numpy as np
import torch 
import gpytorch

class ExactGPyTorchRegressionModel(gpytorch.models.ExactGP):
    allowed_likelihood_types = (
        gpytorch.likelihoods.GaussianLikelihood,
        gpytorch.likelihoods.GaussianLikelihoodWithMissingObs,
        gpytorch.likelihoods.FixedNoiseGaussianLikelihood)
    def __init__(self, x_t, y_t, prior_mean, prior_cov, likelihood, use_gpu=False):
        if isinstance(x_t,np.ndarray): x_t = torch.from_numpy(x_t)
        if isinstance(y_t,np.ndarray): y_t = torch.from_numpy(y_t)
        assert x_t.ndim==2 and y_t.ndim==1 and len(x_t)==len(y_t)
        super(ExactGPyTorchRegressionModel, self).__init__(x_t,y_t,likelihood)
        assert isinstance(self.likelihood,ExactGPyTorchRegressionModel.allowed_likelihood_types)
        self.mean_module,self.covar_module = prior_mean,prior_cov
        self.d = x_t.shape[1]
        self.use_gpu = use_gpu
        if self.use_gpu:
            assert torch.cuda.is_available()
            self = self.cuda()
            self.likelihood = self.likelihood.cuda()
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)
    def fit(self, optimizer, mll, training_iter, verbose=0):
        self.train()
        self.likelihood.train()
        if verbose: print('\tgpytorch model fitting')
        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.__call__(self.train_inputs[0])
            loss = -mll(output,self.train_targets)
            loss.backward()
            if verbose and (i+1)%verbose==0:
                print('\t\titer %-3d of %d'%(i+1,training_iter))
                for name,val in self.named_parameters(): print('\t\t\t%s %.2e'%(name.ljust(50,'.'),val))
            optimizer.step()
    def predict(self, x, noise_const=0, chunk_size=2**15):
        if isinstance(x,np.ndarray): x = torch.from_numpy(x)
        assert x.ndim==2 and x.shape[1]==self.d
        self.eval()
        self.likelihood.eval()
        n = len(x)
        mean_post,std_post = np.zeros(n,dtype=float),np.zeros(n,dtype=float)
        for i in range(0,n,chunk_size):
            lchunk,uchunk = i,min(i+chunk_size,n)
            noise_chunk = noise_const*torch.ones(uchunk-lchunk)
            mean_post[lchunk:uchunk],std_post[lchunk:uchunk] = self._predict_batch(x[lchunk:uchunk],noise_chunk)
        return mean_post,std_post
    def _predict_batch(self, x, noise):
        if self.use_gpu: x,noise = x.cuda(),noise.cuda()
        with torch.no_grad(): observed_pred = self.likelihood(self.__call__(x),noise=noise)
        mean_post = observed_pred.mean
        std_post = observed_pred.stddev
        if self.use_gpu: mean_post,std_post = mean_post.cpu(),std_post.cpu()
        if self.use_gpu: del x,noise,observed_pred; torch.cuda.empty_cache()
        return mean_post.numpy(),std_post.numpy()
    def add_data(self, x_t_new, y_t_new):
        if isinstance(x_t_new,np.ndarray): x_t_new = torch.from_numpy(x_t_new)
        if isinstance(y_t_new,np.ndarray): y_t_new = torch.from_numpy(y_t_new)
        assert x_t_new.ndim==2 and x_t_new.shape[1]==self.d and y_t_new.ndim==1 and len(x_t_new)==len(y_t_new)
        if self.use_gpu: x_t_new,y_t_new = x_t_new.cuda(),y_t_new.cuda()
        fantasy_model = self.get_fantasy_model(x_t_new,y_t_new)
        if self.use_gpu: del x_t_new,y_t_new; torch.cuda.empty_cache()
        return fantasy_model