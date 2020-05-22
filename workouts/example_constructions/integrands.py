""" 
Sample Integrand objects and usage
python workouts/example_constructions/integrands.py > outputs/example_constructions/integrands.log
"""

from qmcpy import *
from numpy import *


def integrands(n=2**15):
    bar = '\n'+'~'*100+'\n'
    print(bar)

    # Asian Call (Single Level)
    distribution = IIDStdUniform(dimension=4, seed=7)
    measure = BrownianMotion(distribution)
    integrand = AsianCall(
        measure = measure,
        volatility = 0.5,
        start_price = 30,
        strike_price = 25,
        interest_rate = 0,
        mean_type = 'arithmetic')
    samples = distribution.gen_samples(n=n)
    y = integrand.f(samples)
    print(integrand)
    print('Asian Call (Single Level) approx with %d samples: %.3f\n%s'%(n,y.mean(),bar))

    # Asian Call (Multi-Level)
    distribution = Lattice(seed=7)
    measure = BrownianMotion(distribution)
    integrand = AsianCall(measure,
        volatility = 0.5,
        start_price = 30,
        strike_price = 25,
        interest_rate = 0,
        mean_type = 'arithmetic',
        multi_level_dimensions = [4,16,64])
    print(integrand)
    print('Asian Call (Multi Level) passing')

    # Keister
    distribution = IIDStdGaussian(dimension=3, seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    samples = distribution.gen_samples(n=n)
    y = integrand.f(samples)
    print(integrand)
    print('Keister approx with %d samples: %.3f\n%s'%(n,y.mean(),bar))

    # Linear
    distribution = Lattice(dimension=2, scramble=True, seed=7, backend='GAIL')
    measure = Uniform(distribution)
    integrand = Linear(measure)
    samples = distribution.gen_samples(n_min=0,n_max=n)
    y = integrand.f(samples)
    print(integrand)
    print('Linear approx with %d samples: %.3f\n%s'%(n,y.mean(),bar))

    # Quick Construct
    distribution = Sobol(dimension=3, scramble=True, seed=7, backend='QRNG')
    measure = Lebesgue(distribution,lower_bound=[1,2,3],upper_bound=7)
    integrand = QuickConstruct(measure,lambda x: x[:,0]*x[:,1]**x[:,2])
    samples = distribution.gen_samples(n_min=0,n_max=n)
    y = integrand.f(samples)
    print(integrand)
    print('QuickConstruct approx with %d samples: %.3f\n%s'%(n,y.mean(),bar))


if __name__ == '__main__':
    integrands(n=2**15)