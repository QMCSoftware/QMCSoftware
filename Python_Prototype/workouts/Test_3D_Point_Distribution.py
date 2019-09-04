import matplotlib.pyplot as mpl_plt
from numpy import random,linspace,meshgrid, zeros,arange
from mpl_toolkits.mplot3d.axes3d import Axes3D

random.seed(7)
from scipy.stats import norm

from third_party.latticeseq_b2 import get_RS_lattice_b2
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.measure.measure import measure
from algorithms.function.KeisterFun import KeisterFun

dim = 2

fun = KeisterFun()
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[1/2])
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[dim]))
fun = fun.transformVariable(measureObj, distribObj)

j = 3
colors = ['r','b','g']
n = 32
var = 1/2
coordIdx = arange(1,dim+1)

# Examples
'''
# CLT_Rep Example
stopObj = CLT_Rep(nInit=n,nMax=2**15,absTol=.01,J=j)
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[var])
distribObj = Mesh(trueD=measure().mesh(dimension=[dim],meshType='lattice'))
sol,out = integrate(fun,measureObj,distribObj,stopObj)
#print(sol,out)

# CLT Example
stopObj = CLTStopping(nInit=16,absTol=.3,alpha=.01,inflate=1.2)
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[1/2])
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[dim])) # IID sampling
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)
print(sol,out)
'''
# Based on running the above CLT Example
eps_list = [.5,.4,.3]
n_list = [58,81,131]
muHat_list = [1.93,1.91,1.94]

# Function Points
nx, ny = (100, 100)
points_fun = zeros((nx*ny,3))
x = linspace(-3, 3, nx)
y = linspace(-3, 3, ny)
x_2d,y_2d = meshgrid(x,y)
points_fun[:,0] = x_2d.flatten()
points_fun[:,1] = y_2d.flatten()
points_fun[:,2] = fun.f(points_fun[:,:2],coordIdx)
x_surf = points_fun[:,0].reshape((nx,ny))
y_surf = points_fun[:,1].reshape((nx,ny))
z_surf = points_fun[:,2].reshape((nx,ny))

# 3D Plot
fig = mpl_plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131,projection='3d')
ax2 = fig.add_subplot(132,projection='3d')
ax3 = fig.add_subplot(133,projection='3d')

plotType = 'pregen' #'randn' 'lattice'
#plotType = 'lattice'
for idx,ax in enumerate([ax1,ax2,ax3]):
    # Surface
    ax.plot_surface(x_surf,y_surf,z_surf,cmap='winter',alpha=.2)
    # Scatters
    points_distrib = zeros((n,dim+1))
    if plotType == 'lattice':
        muhat = zeros(j)
        set_x = get_RS_lattice_b2(n,dim,j) 
        for i,xu in enumerate(set_x):
            points_distrib[:,:2] = xu
            points_distrib[:,2] = fun.f(points_distrib[:,:2],coordIdx)
            ax.scatter(points_distrib[:,0],points_distrib[:,1],points_distrib[:,2],color=colors[i],s=15)
            muhat[i] = points_distrib[:,2].mean()
        mu = muhat.mean()
        epsilon = muhat.std()
    if plotType == 'randn':
        points_distrib[:,:2] = random.randn(n,dim)
        points_distrib[:,2] = fun.f(points_distrib[:,:2],coordIdx)
        ax.scatter(points_distrib[:,0],points_distrib[:,1],points_distrib[:,2],color='r',s=5)
        mu = points_distrib[:,2].mean()
        std2 = points_distrib[:,2].std()
        epsilon = -norm.ppf(.01/2) * 1.2 * (std2**2/n).sum(0)**.5
    if plotType == 'pregen':
        points_distrib[:,:2] = random.randn(n,dim)
        points_distrib[:,2] = fun.f(points_distrib[:,:2],coordIdx)
        ax.scatter(points_distrib[:,0],points_distrib[:,1],points_distrib[:,2],color='r',s=5)
        n = n_list[idx]
        epsilon = eps_list[idx]
        mu = muHat_list[idx]
        ax.scatter(points_distrib[:,0],points_distrib[:,1],points_distrib[:,2],color='r',s=5)
    ax.set_title('\t$\epsilon$ = %-7.1f $n$ = %-7d $\hat{\mu}$ = %-7.2f '%(epsilon,n,mu),fontdict={'fontsize': 14})
    # axis metas
    n *= 2
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.set_xlabel('$x_1$',fontdict={'fontsize': 14})
    ax.set_ylabel('$x_2$',fontdict={'fontsize': 14})
    ax.set_zlabel('$f\:(x_1,x_2)$',fontdict={'fontsize': 14})
    ax.view_init(20,45)
    
# Output
mpl_plt.savefig('Outputs/Three_3d_SurfaceScatters.png',
        dpi = 500,
        bbox_inches = 'tight',
        pad_inches = .15)
mpl_plt.show()
