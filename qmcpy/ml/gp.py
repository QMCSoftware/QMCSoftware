import gpytorch 
import torch 

class IndepVecVGP(gpytorch.models.ApproximateGP):
    """
    https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html
    """
    def __init__(self, num_tasks, num_inducing_pts, dimension, mean_module=None, covar_module=None):
        inducing_points = torch.rand(num_tasks,num_inducing_pts,dimension)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_pts,batch_shape=torch.Size([num_tasks]))
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(self,inducing_points,variational_distribution,learn_inducing_locations=True),
            num_tasks=num_tasks)
        super().__init__(variational_strategy)
        if mean_module is None:
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        else:
            self.mean_module = mean_module
        if covar_module is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RQKernel(batch_shape=torch.Size([num_tasks]),ard_num_dims=dimension),
                batch_shape=torch.Size([num_tasks]))
        else:
            covar_module = covar_module
        self.num_tasks = num_tasks
        self.dimension = dimension
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)

# class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super().__init__(train_x, train_y, likelihood)
#         #self.mean_module = gpytorch.means.LinearMean(input_size=2*nxticks,batch_shape=torch.Size([nxticks]))
#         self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([(1+(nxticks+2))*(nxticks+2)//2]))
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(
#                 ard_num_dims = None, 
#                 #ard_num_dims = 2*nxticks, 
#                 batch_shape=torch.Size([(1+(nxticks+2))*(nxticks+2)//2])),
#             batch_shape=torch.Size([(1+(nxticks+2))*(nxticks+2)//2]))
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
#             gpytorch.distributions.MultivariateNormal(mean_x, covar_x))

# class MultitaskGPModel(gpytorch.models.ApproximateGP):
#     def __init__(self, num_inducing_pts, num_tasks, num_latents, num_inputs):
#         inducing_points = torch.rand(num_latents, num_inducing_pts, num_inputs)
#         variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_pts,batch_shape=torch.Size([num_latents]))
#         variational_strategy = gpytorch.variational.LMCVariationalStrategy(
#             gpytorch.variational.VariationalStrategy(
#                 self, inducing_points, variational_distribution, learn_inducing_locations=True),
#             num_tasks=num_tasks,
#             num_latents=num_latents,
#             latent_dim=-1)
#         super().__init__(variational_strategy)
#         self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
#             batch_shape=torch.Size([num_latents]))
#         self.num_tasks = num_tasks
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)