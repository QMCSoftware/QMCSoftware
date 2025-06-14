from .geometric_brownian_motion_copy import GeometricBrownianMotion
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..discrete_distribution import DigitalNetB2

sampler = DigitalNetB2(dimension=4, seed=42)


gbm = GeometricBrownianMotion(
    sampler=sampler,
    t_final=2,           
    initial_value=1,     
    drift=0.1,           
    diffusion=0.2,       
    decomp_type='PCA'    
)


samples = gbm.gen_samples(2**2)  
print("Generated Samples:")
print(samples)


print("\nGeometricBrownianMotion Summary:")
print(gbm)