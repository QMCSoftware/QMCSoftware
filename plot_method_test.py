from qmcpy import Lattice
import matplotlib.pyplot as plt


lat = Lattice(3)
x, y = lat.plot(64,0,1,color="green")
plt.show()



