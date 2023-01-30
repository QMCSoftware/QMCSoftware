import matplotlib.pyplot as plt

from qmcpy import Lattice

lat = Lattice(3)
lat.plot(64,1,2)
plt.show()
