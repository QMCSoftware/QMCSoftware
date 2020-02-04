from qmcpy import Lattice, Sobol, IIDStdGaussian, IIDStdUniform
from numpy import *
set_printoptions(threshold=1e10)

d = 2
r = 0
scramble = True
seed = 7
bar = '~'*100+'\n'

# IID RNG
iid_std_uniform = IIDStdUniform(dimension=d, seed=seed)
iid_std_gaussian = IIDStdGaussian(dimension=d, seed=seed)
print(bar)
n = 4
for iid_rng_obj in [iid_std_uniform,iid_std_gaussian]:
    x = iid_rng_obj.gen_samples(n=n)
    print(iid_rng_obj)
    print('Shape:',x.shape)
    print('Nodes:\n\t%s'%str(x).replace('\n','\n\t'))
    print(bar)

# Quasi RNG
lat_obj = Lattice(dimension=d, scramble=scramble, replications=r, seed=seed, backend='GAIL')
sob_obj = Sobol(dimension=d, scramble=scramble, replications=r, seed=seed, backend='PyTorch')
n_min = 4
n_max = 8
for qrng_obj in [lat_obj,sob_obj]:
    x = qrng_obj.gen_samples(n_min=n_min, n_max=n_max)
    print(qrng_obj)
    print('Shape:',x.shape)
    print('Nodes:\n\t%s'%str(x).replace('\n','\n\t'))
    print(bar)