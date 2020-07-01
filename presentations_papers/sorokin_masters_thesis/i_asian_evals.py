from qmcpy import *
# single level 
dd = Sobol(4,seed=7)
m = BrownianMotion(dd)
ac = AsianCall(m)
x = dd.gen_samples(2**10)
y = ac.f(x)
est1 = y.mean()
# multilevel 
dd2 = Sobol(seed=7)
m2 = BrownianMotion(dd2,drift=1)
level_dims = [2,4,8]
ac2 = AsianCall(m2,multi_level_dimensions=level_dims)
est2 = 0
for l in range(len(level_dims)):
    new_dim = ac2.dim_at_level(l)
    m2.set_dimension(new_dim)
    x2 = dd2.gen_samples(2**10)
    y2 = ac2.f(x2,l=l)
    est2 += y2.mean()