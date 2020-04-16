"""
Sample DiscreteDistribution objects and usage
python workouts/example_constructions/distributions.py > outputs/example_constructions/distributions.log
"""

from qmcpy import Lattice, Sobol, IIDStdGaussian, IIDStdUniform
from numpy import *
set_printoptions(threshold=1e10)


def distributions(n=4, dimension=2, replications=0, scramble=True, seed=7):
    bar = '\n'+'~'*100+'\n'
    print(bar)

    # IID RNG
    iid_std_uniform = IIDStdUniform(
        dimension = dimension,
        seed = seed)
    iid_std_gaussian = IIDStdGaussian(
        dimension = dimension,
        seed = seed)
    for iid_rng_obj in [iid_std_uniform,iid_std_gaussian]:
        x = iid_rng_obj.gen_samples(n=n)
        print(iid_rng_obj)
        print('Shape:',x.shape)
        print('Nodes:\n\t%s'%str(x).replace('\n','\n\t'))
        print(bar)

    # Quasi RNG
    lat_obj = Lattice(
        dimension = dimension,
        scramble = scramble,
        replications = replications,
        seed = seed,
        backend = 'GAIL')
    sob_obj = Sobol(
        dimension = dimension,
        scramble = scramble,
        replications = replications,
        seed = seed,
        backend = 'PyTorch')
    for qrng_obj in [lat_obj,sob_obj]:
        x = qrng_obj.gen_samples(n_min=0, n_max=n)
        print(qrng_obj)
        print('Shape:',x.shape)
        print('Nodes:\n\t%s'%str(x).replace('\n','\n\t'))
        print(bar)


if __name__ == '__main__':
    distributions(n=4, dimension=2, replications=0, scramble=True, seed=7)