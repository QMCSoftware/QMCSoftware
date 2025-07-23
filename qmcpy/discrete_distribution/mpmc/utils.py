import torch
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def L2dis(x):
    N = x.size(1) 
    dim = x.size(2)
    prod1 = 1. - x ** 2.
    prod1 = torch.prod(prod1, dim=2) #multiplying across second dimenstion of x (dim)
    sum1 = torch.sum(prod1, dim=1) #summing across second dimension of x (number of points in each batch)
    pairwise_max = torch.maximum(x[:, :, None, :], x[:, None, :, :])
    product = torch.prod(1 - pairwise_max, dim=3)
    sum2 = torch.sum(product, dim=(1, 2))
    one_dive_N = 1. / N
    out = torch.sqrt(
        math.pow(3., -dim) 
        - one_dive_N * math.pow(2., 1. - dim) * sum1 
        + 1. / math.pow(N, 2.) * sum2)
    return out


def L2dis_weighted(x, gamma):
    N = x.size(1) 
    dim = x.size(2)

    p1 = torch.prod(1 + gamma/3)
    prod1 = 1 + (gamma[None, None, :]/2)*(1. - x ** 2.)
    prod1 = torch.prod(prod1, dim=2) #multiplying across second dimenstion of x (dim)
    sum1 = torch.sum(prod1, dim=1) #summing across second dimension of x (number of points in each batch)
    
    pairwise_max = torch.maximum(x[:, :, None, :], x[:, None, :, :])
    product = torch.prod(1 + gamma[None, None, None, :]*(1- pairwise_max), dim=3)
    sum2 = torch.sum(product, dim=(1, 2))

    out = torch.sqrt(
        p1
        - 2./N * sum1 
        + 1. / math.pow(N, 2.) * sum2)
    
    return out



def L2ctr(x):
    N = x.size(1)
    dim = x.size(2)

    prod1 = 0.5*(torch.abs(x - 0.5) - (torch.abs(x - 0.5))**2.)
    prod1 = torch.prod(prod1, dim=2)
    sum1 = torch.sum(prod1, dim=1)

    prod2 = 0.5*(torch.abs(x[:, :, None, :] - 0.5) 
                    + torch.abs(x[:, None, :, :] - 0.5) 
                    - torch.abs(x[:, :, None, :] - x[:, None, :, :]))
    product = torch.prod(prod2, dim=3)
    sum2 = torch.sum(product, dim=(1, 2))
    two_div_N = 2. / N
    
    out = math.pow(12., -dim) - two_div_N * sum1 + 1. / math.pow(N, 2.) * sum2
    return out



def L2ctr_weighted (x, gamma):
    N = x.size(1)
    dim = x.size(2)

    term1_val = torch.prod(1.0 + gamma**2 / 12.0)

    abs_x_half = torch.abs(x - 0.5)
    prod1 = 1.0 + (gamma.view(1, 1, dim) / 2.0) * (abs_x_half - abs_x_half**2)
    prod1 = torch.prod(prod1, dim=2)
    sum1 = torch.sum(prod1, dim=1)

    x_i = x.unsqueeze(2)
    x_j = x.unsqueeze(1)

    prod2 = 1.0 + (gamma.view(1, 1, 1, dim) / 2.0) * (torch.abs(x_i - 0.5) + torch.abs(x_j - 0.5) - torch.abs(x_i - x_j))
    prod2 = torch.prod(prod2, dim=3)
    sum2 = torch.sum(prod2, dim=(1, 2))

    out_sq = term1_val - (2.0 / N) * sum1 + (1.0 / (N**2)) * sum2
    out = torch.sqrt(torch.relu(out_sq))

    return out



def L2ext(x):
    N = x.size(1)
    dim = x.size(2)

    prod1 = 0.5*(x - x**2.)
    prod1 = torch.prod(prod1, dim = 2)
    sum1 = torch.sum(prod1, dim = 1)

    prod2 = torch.min(x[: ,: ,None ,: ], x[: ,None ,: ,: ]) - x[: ,: ,None ,: ] * x[: ,None ,: ,: ]
    product = torch.prod(prod2, dim = 3)
    sum2 = torch.sum(product, dim = (1,2))

    out = math.pow(12., -dim) - (2. / N) * sum1 + math.pow(N, - 2.) * sum2
    return out


def L2ext_weighted(x, gamma):
    N = x.size(1)
    dim = x.size(2)

    p1 = torch.prod(1 + gamma/12)

    prod1 = 1 + gamma[None, None, :] * 0.5*(x - x**2.)
    prod1 = torch.prod(prod1, dim = 2)
    sum1 = torch.sum(prod1, dim = 1)

    prod2 = 1 + gamma[None, None, None, :] * torch.min(x[: ,: ,None ,: ], x[: ,None ,: ,: ]) - x[: ,: ,None ,: ] * x[: ,None ,: ,: ]
    product = torch.prod(prod2, dim = 3)
    sum2 = torch.sum(product, dim = (1,2))

    out = p1 - (2. / N) * sum1 + math.pow(N, - 2.) * sum2
    return out

def L2per(x):
    N = x.size(1)
    dim = x.size(2)

    prod2 = 0.5 - torch.abs(x[: ,: ,None ,: ] - x[: ,None ,: ,: ]) + (x[: ,: ,None ,: ] - x[: ,None ,: ,: ])**2
    product = torch.prod(prod2, dim = 3)
    sum2 = torch.sum(product, dim = (1,2))

    out = - math.pow(3., -dim) + math.pow(N, - 2.) * sum2
    return out


def L2per_weighted(x, gamma):
    N = x.size(1)
    dim = x.size(2)

    p1 = torch.prod(1 + gamma/3)

    prod2 = 1 + gamma[None, None, None, :] * 0.5 - torch.abs(x[: ,: ,None ,: ] - x[: ,None ,: ,: ]) + (x[: ,: ,None ,: ] - x[: ,None ,: ,: ])**2
    product = torch.prod(prod2, dim = 3)
    sum2 = torch.sum(product, dim = (1,2))

    out = - p1 + math.pow(N, - 2.) * sum2
    return out


def L2sym(x): 
    N = x.size(1)
    dim = x.size(2)

    prod1 = 0.5*(x - x**2.)
    prod1 = torch.prod(prod1, dim = 2)
    sum1 = torch.sum(prod1, dim = 1)

    prod2 = 0.25 * (1 - 2 * torch.abs(x[: ,: ,None ,: ] - x[: ,None ,: ,: ]))
    product = torch.prod(prod2, dim = 3)
    sum2 = torch.sum(product, dim = (1,2))

    out = math.pow(12., -dim) - (2. / N) * sum1 + math.pow(N, - 2.) * sum2
    return out


def L2sym_weighted(x, gamma): 
    N = x.size(1)
    dim = x.size(2)

    p1 = torch.prod(1 + gamma/12)

    prod1 = 1 + gamma[None, None, :] * 0.5*(x - x**2.)
    prod1 = torch.prod(prod1, dim = 2)
    sum1 = torch.sum(prod1, dim = 1)

    prod2 = 1 + gamma[None, None, None, :] * 0.25 * (1 - 2 * torch.abs(x[: ,: ,None ,: ] - x[: ,None ,: ,: ]))
    product = torch.prod(prod2, dim = 3)
    sum2 = torch.sum(product, dim = (1,2))

    out = p1 - (2. / N) * sum1 + math.pow(N, - 2.) * sum2
    return out
    


def L2ags (x):
    N = x.size(1)
    dim = x.size(2)

    term1 = ((1 / 3) ** dim)

    sum1 = 2. * x - 2. * x**2
    sum2 = 1.0 + sum1 
    prod1 = sum2 / 4.
    prod2 = torch.prod(prod1, dim=2)
    sum2 = torch.sum(prod2, dim=1)
    term2 = -(2.0 / N) * sum2

    x_i = x.unsqueeze(2) 
    x_j = x.unsqueeze(1) 
    sum3 = torch.abs(x_i - x_j)
    prod3 = torch.prod( (1.0 - sum3) / 2.0, dim=3)
    term3_sum = torch.sum(prod3, dim=(1, 2))
    term3 = (1.0 / (N * N)) * term3_sum
    
    out = torch.sqrt(term1+ term2 + term3)
    return out






def L2ags_weighted (x, gamma):
    N = x.size(1)
    dim = x.size(2)

    term1 = torch.prod(1.0 + gamma/ 3.0)

    g_term2 = gamma.view(1, 1, dim)
    sum1 = g_term2 * (1 + 2. * x - 2. * x**2)
    prod1 = 1 + sum1 / 4. 
    prod2 = torch.prod(prod1, dim=2)
    sum_prod2 = torch.sum(prod2, dim=1)
    term2 = -(2. / N) * sum_prod2

    x_i = x.unsqueeze(2)
    x_j = x.unsqueeze(1)
    g_term3 = gamma.view(1, 1, 1, dim)   # need to reshape?
    sum3 = g_term3 * (1 - torch.abs(x_i - x_j))
    prod3 = torch.prod(1 + sum3/ 2., dim=3)
    term3_sum = torch.sum(prod3, dim=(1, 2))
    term3 = (1. / (N * N)) * term3_sum

    out_sq = term1 + term2 + term3
    out = torch.sqrt(out_sq)
    
    return out


def L2mix(x):
     N = x.size(1)
     dim = x.size(2)
     prod1 = 2/3 - 1/4 * (torch.abs(x - 1/2)) - 1/4 * ((x - 1/2)**2)
     prod1 = torch.prod(prod1, dim = 2)
     sum1 = torch.sum(prod1, dim = 1)

     prod2 = 7/8 - 1/4 * torch.abs(x[: ,: ,None ,: ] - 1/2) - 1/4 * torch.abs(x[:, None, :, :] - 1/2) - 3/4*torch.abs(x[: ,: ,None ,: ] - x[:, None, :, :]) + 1/2 * torch.pow(x[: ,: ,None ,: ] - x[:, None, :, :], 2)
     product = torch.prod(prod2, dim = 3)
     sum2 = torch.sum(product, dim = (1,2))


     out = torch.sqrt(abs(math.pow(7./12., dim) - (2. / N) * sum1 + math.pow(N, -2.) * sum2))
     return out





def L2mix_weighted(x, gamma):
    N = x.size(1)
    dim = x.size(2)

    term1_val = torch.prod(1.0 + 7 * gamma / 12.0)

    gamma_r = gamma.view(1, 1, dim)
    inner_sum1 = 1.0 + gamma_r * (2./3. - 0.25 * torch.abs(x - 0.5) - 0.25 * (x - 0.5)**2)
    prod1 = torch.prod(inner_sum1, dim=2)
    sum1 = torch.sum(prod1, dim=1)
    term2_val = (2.0 / N) * sum1

    
    x_i = x.unsqueeze(2)  #reshape to (batch, N, 1, dim)
    x_j = x.unsqueeze(1)  # reshape to (batch, 1, N, dim)
    
    gamma_r_term3 = gamma.view(1, 1, 1, dim) 

    inner_sum2 = 1.0 + gamma_r_term3 * (
        7./8.
        - 0.25 * torch.abs(x_i - 0.5)
        - 0.25 * torch.abs(x_j - 0.5)
        - 0.75 * torch.abs(x_i - x_j)
        + 0.5 * (x_i - x_j)**2
    )
    prod2 = torch.prod(inner_sum2, dim=3)
    sum2 = torch.sum(prod2, dim=(1, 2))
    term3_val = (1.0 / (N * N)) * sum2

    out_sq = term1_val - term2_val + term3_val

    out = torch.sqrt((out_sq))
    
    return out