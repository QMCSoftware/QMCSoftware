>>> from qmcpy import *
>>> CubLattice_g(
...    Keister(
...       Gaussian(
...          Lattice(dimension=2),
...          covariance = 1./2)),
...    abs_tol = .01).integrate()