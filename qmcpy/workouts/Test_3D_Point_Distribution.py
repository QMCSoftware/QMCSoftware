from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as mpl_plt
from scipy.stats import norm
from numpy import arange, linspace, meshgrid, random, zeros
random.seed(7)

from workouts import summary_qmc
from algorithms.distribution.Measures import IIDZeroMeanGaussian,StdGaussian,Lattice
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.integrand.KeisterFun import KeisterFun

dim = 2
j = 3
colors = ['r','b','g']
n = 32
var = 1/2
coordIdx = arange(1,dim+1)

fun = KeisterFun()
measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
distribObj = IIDDistribution(trueD=StdGaussian(dimension=[dim]), rngSeed=7)
fun = fun.transform_variable(measureObj, distribObj)

# Examples for generating 'pregen' figure's constants
'''
import sys
from algorithms.stop.CLTRep import CLTRep
from algorithms.stop.CLTStopping import CLTStopping
from algorithms.integrate import integrate
#     CLTRep Example
funObj = KeisterFun()
distribObj = QuasiRandom(trueD=Lattice(dimension=[dim]),rngSeed=7)
stopObj = CLTRep(distribObj,nInit=n,nMax=2**15,absTol=.01,J=j)
measureObj = IIDZeroMeanGaussian(dimension=[dim],variance=[var])
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)
#     CLT Example
funObj = KeisterFun()
distribObj = IIDDistribution(trueD=StdGaussian(dimension=[dim]),rngSeed=7)
stopObj = CLTStopping(distribObj,nInit=16,absTol=.3,alpha=.01,inflate=1.2)
measureObj = IIDZeroMeanGaussian(dimension=[dim],variance=[1/2])
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)
sys.exit(0)
'''

# 'pregen' constants based on running the above CLT Example
eps_list = [.5,.4,.3]
n_list = [50,68,109]
muHat_list = [1.876,1.806,1.883]

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
        set_x = QuasiRandom().get_RS_lattice_b2(n,dim,j) 
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
mpl_plt.savefig('outputs/Three_3d_SurfaceScatters.png',
        dpi = 500,
        bbox_inches = 'tight',
        pad_inches = .15)
mpl_plt.show()
