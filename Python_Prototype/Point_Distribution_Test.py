from matplotlib import pyplot as mpl_plot
from numpy import linspace,random
random.seed(82)

n = 75
# Scatter
mpl_plot.scatter(random.rand(n),random.rand(n),color='b',label='Uniform Distribution')
# f(x) = x^2
x = linspace(0,1,1000)
mpl_plot.plot(x,x**2,color='r',label='f(x)=x\u00B2')

# Meta
mpl_plot.xlim([0,1])
mpl_plot.ylim([0,1])
# mpl_plot.legend(loc=2,prop={'size': 15},framealpha=1)

# Output
mpl_plot.savefig('DevelopOnly/Outputs/Uniform_Distrib_ScatterLilePlot.png',
    dpi=200,
    bbox_inches = 'tight',
    pad_inches = .05)
mpl_plot.show()