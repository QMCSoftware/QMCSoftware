from torch.quasirandom import SobolEngine
from scipy.stats import norm

sobol = SobolEngine(dimension=64, scramble=True, seed=8)
x1 = sobol.draw(32)
x2 = sobol.draw(32)
x3 = sobol.draw(64)
x4 = sobol.draw(128)
# Pytorch Sobol returns 1
element1 = x4.numpy()[11, 34]
inv1 = norm.ppf(element1)
print(element1)
print(inv1)
# Replace 1 with something close to 1
element2 = 1 - 1e-15
inv2 = norm.ppf(element2)
print(element2)
print(inv2)

print(norm.cdf(10))
