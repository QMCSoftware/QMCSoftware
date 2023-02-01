import matplotlib.pyplot as plt

from qmcpy import Lattice

lat = Lattice(dimension=3,seed=7)
lat.plot(2**7, 0, 1,color = "green")
lat.plot(2**7, 1, 2)
plt.show()
