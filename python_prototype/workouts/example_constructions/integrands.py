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
    measure = BrownianMotion(distribution, time_vector=[1/4,1/2,3/4,1])
    integrand = AsianCall(
        measure = measure,
        volatility = 0.5,
        start_price = 30,
        strike_price = 25,
        interest_rate = 0,
        mean_type = 'arithmetic')
    samples = measure.gen_samples(n=n)
    y = integrand.f(samples)
    print(integrand)
    print('Asian Call (Single Level) approx with %d samples: %.3f\n%s'%(n,y.mean(),bar))

    # Asian Call (Multi-Level)
    time_vector = [
        arange(1/4,5/4,1/4),
        arange(1/16,17/16,1/16),
        arange(1/64,65/64,1/64)]
    levels = len(time_vector)
    distributions = MultiLevelConstructor(levels,
        Lattice,
            dimension = [len(tv) for tv in time_vector],
            scramble = True,
            replications=0,
            seed = 7,
            backend='GAIL')
    measures = MultiLevelConstructor(levels,
        BrownianMotion,
            distribution = distributions,
            time_vector = time_vector)
    integrands = MultiLevelConstructor(levels,
        AsianCall,
            measure = measures,
            volatility = 0.5,
            start_price = 30,
            strike_price = 25,
            interest_rate = 0,
            mean_type = 'arithmetic')
    y = 0
    for l in range(levels):
        samples_l = measures[l].gen_samples(n_min=0,n_max=n)
        y += integrands[l].f(samples_l)
    print(integrands)
    print('Asian Call (Single Level) approx with %d samples: %.3f\n%s'%(n,y.mean(),bar))

    # Keister
    distribution = IIDStdGaussian(dimension=3, seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    samples = measure.gen_samples(n=n)
    y = integrand.f(samples)
    print(integrand)
    print('Keister approx with %d samples: %.3f\n%s'%(n,y.mean(),bar))

    # Linear
    distribution = Lattice(dimension=2, scramble=True, replications=0, seed=7, backend='GAIL')
    measure = Uniform(distribution)
    integrand = Linear(measure)
    samples = measure.gen_samples(n_min=0,n_max=n)
    y = integrand.f(samples)
    print(integrand)
    print('Linear approx with %d samples: %.3f\n%s'%(n,y.mean(),bar))

    # Quick Construct
    distribution = Sobol(dimension=3, scramble=True, replications=0, seed=7, backend='MPS')
    measure = Lebesgue(distribution,lower_bound=[1,2,3],upper_bound=7)
    integrand = QuickConstruct(measure,lambda x: x[:,0]*x[:,1]**x[:,2])
    samples = measure.gen_samples(n_min=0,n_max=n)
    y = integrand.f(samples)
    print(integrand)
    print('QuickConstruct approx with %d samples: %.3f\n%s'%(n,y.mean(),bar))


if __name__ == '__main__':
    integrands(n=2**15)