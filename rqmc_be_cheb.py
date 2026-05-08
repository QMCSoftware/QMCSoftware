import numpy as np 
from matplotlib import pyplot 
pyplot.style.use("seaborn-v0_8-whitegrid")
DEFAULTFONTSIZE = 30
pyplot.rcParams['xtick.labelsize'] = DEFAULTFONTSIZE
pyplot.rcParams['ytick.labelsize'] = DEFAULTFONTSIZE
pyplot.rcParams['ytick.labelsize'] = DEFAULTFONTSIZE
pyplot.rcParams['axes.titlesize'] = DEFAULTFONTSIZE
pyplot.rcParams['figure.titlesize'] = DEFAULTFONTSIZE
pyplot.rcParams["axes.labelsize"] = DEFAULTFONTSIZE
pyplot.rcParams['legend.fontsize'] = DEFAULTFONTSIZE
pyplot.rcParams['font.size'] = DEFAULTFONTSIZE
pyplot.rcParams['lines.linewidth'] = 5
pyplot.rcParams['lines.markersize'] = 15
PW = 30 # inches

R1 = 1000 
gamma = 1 
alpha2 = 0.025
p = 2 
m1 = 0 
r = 3 
m2 = np.arange(1,21)
assert R1>0 
assert gamma>0 
assert 0<alpha2<1 
assert p>=2 
assert m1>=0 
assert (m2>=0).all()
assert (m1<m2).all()

epsilon_cheb = np.sqrt(gamma/(alpha2*R1)*(float(p)**((1-r)*m1)-float(p)**((1-r)*m2))/(p**m2-p**m1))

n1 = p**m1
n2 = p**m2

fig,ax = pyplot.subplots(nrows=1,ncols=1,figsize=(PW/1.5,PW/2))
ax.plot(n2,epsilon_cheb,'-o')
ax.set_xscale('log',base=p)
ax.set_yscale('log',base=10)
ax.set_xlabel(r'$n_2$')
ax.set_ylabel(r'$\varepsilon$')
ax.set_title("Chebychev")
fig.suptitle(r"If $\varepsilon$ below the line then RQMC with $n_2$ outperforms RQMC with $n_1 = %d$"%n1)
fig.savefig("rqmc_be_cheb.png",dpi=256,bbox_inches='tight')