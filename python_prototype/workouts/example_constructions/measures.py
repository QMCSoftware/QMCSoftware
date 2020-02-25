"""
Sample TrueMeasure objects and usage
python workouts/example_constructions/measures.py > outputs/example_constructions/measures.log
"""

from qmcpy import *
from copy import deepcopy
from numpy import *
set_printoptions(threshold=1e10)


def measures(n=2, dimension=2, replications=0, seed=7):
    bar = '\n'+'~'*100+'\n'
    print(bar)

    iid_std_u = IIDStdUniform(dimension=dimension, seed=seed)
    iid_std_g = IIDStdGaussian(dimension=dimension, seed=seed)

    # Brownian motion
    time_vector = [.5,1]
    for distribution in [iid_std_u,iid_std_g]:
        measure = BrownianMotion(distribution, time_vector=time_vector)
        x = measure.gen_samples(n)
        print(measure)
        print(bar)

    # Gaussian
    mu = [3, 3]
    sigma = [5, 7]
    for distribution in [iid_std_u,iid_std_g]:
        measure = Gaussian(distribution, mean=mu, covariance=sigma)
        x = measure.gen_samples(n)
        print(measure)
        print(bar)

    # Uniform
    a = [-2, -3]
    b = 4
    for distribution in [iid_std_u,iid_std_g]:
        measure = Uniform(distribution, lower_bound=a, upper_bound=b)
        x = measure.gen_samples(n)
        print(measure)
        print(bar)

    # Lebesgue
    a = [-2, -3]
    b = [2, 4]
    for distribution in [iid_std_u]:
        measure = Lebesgue(distribution, lower_bound=a, upper_bound=b)
        x = measure.gen_samples(n)
        print(measure)
        print(bar)


if __name__ == '__main__':
    measures(n=2, dimension=2, replications=0, seed=7)