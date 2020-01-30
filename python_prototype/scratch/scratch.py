from torch.quasirandom import SobolEngine
from torch import where

print('These issues all concern the implementation of Sobol points \n')
print('The zeroth unscrambled Sobol point should always be zero.')
print('But the implementation here starts with the first point, (0.5, 0.5, ...)')
sobol = SobolEngine(dimension=5, scramble=False)
x = sobol.draw(1)
print(x)
print('Can this be rectified? \n')

print('With probability one, a scrambled Sobol point should never have a coordinate = 1')
print('But with this implementation, it happens far too often')
for ii in range(0,100):
    sobol = SobolEngine(dimension=64, scramble=True, seed=ii)
    x = sobol.draw(2**16)
    where1 = where(x==1)
    if len(where1[0]):
        print('When seed =',ii)
        print('  Sobol point coordinate is 1 at:',where1)
print('Can this be rectified?')

