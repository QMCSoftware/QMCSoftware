import qmcpy as qp
import numpy as np
from math import*


barrier_option = qp.BarrierOption(qp.DigitalNetB2(4,seed=7))

x = barrier_option.discrete_distrib.gen_samples(2**12)
y = barrier_option.f(x)
print(y.mean())
    
#level_dims = [2,4,8]
#barrier_option_multilevel = qp.BarrierOption(qp.DigitalNetB2(seed=7),multilevel_dims=level_dims)
#levels_to_spawn = np.arange(barrier_option_multilevel.max_level+1)
#barrier_option_single_levels = barrier_option_multilevel.spawn(levels_to_spawn)
#yml = 0
#for barrier_option_single_level in barrier_option_single_levels:
        #x = barrier_option_single_level.discrete_distrib.gen_samples(2**12)
        #level_est = barrier_option_single_level.f(x).mean()
        #yml += level_est
#print(yml)