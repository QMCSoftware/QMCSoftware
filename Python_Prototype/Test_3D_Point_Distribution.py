from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as mpl_plt 
from numpy import random,linspace,meshgrid,hstack,zeros,arange,sqrt
random.seed(7)
from scipy.stats import norm

from latticeseq_b2 import get_RS_lattice_b2
from CLT_Rep import CLT_Rep
from CLTStopping import CLTStopping
from IIDDistribution import IIDDistribution
from measure import measure
from Mesh import Mesh
from integrate import integrate
from KeisterFun import KeisterFun
from fun import fun
from LinearFun import LinearFun

fun = KeisterFun()
dim = 2
j = 3
colors = ['r','b','g']
n = 32
var = 1/2
coordIdx = arange(1,dim+1)

# CLT_Rep Example
stopObj = CLT_Rep(nInit=n,nMax=2**15,absTol=.01,J=j)
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[var])
distribObj = Mesh(trueD=measure().mesh(dimension=[dim],meshType='lattice'))
sol,out = integrate(fun,measureObj,distribObj,stopObj)
#print(sol,out)

# CLT Example
stopObj = CLTStopping(nInit=16,absTol=.5,alpha=.01,inflate=1.2)
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[1/2])
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[dim])) # IID sampling
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)
print(sol,out)

# Function Points
nx, ny = (100, 100)
points_fun = zeros((nx*ny,3))
x = linspace(-3, 3, nx)
y = linspace(-3, 3, ny)
x_2d,y_2d = meshgrid(x,y)
points_fun[:,0] = x_2d.flatten()
points_fun[:,1] = y_2d.flatten()
points_fun[:,2] = fun.g(points_fun[:,:2],coordIdx)
x_surf = points_fun[:,0].reshape((nx,ny))
y_surf = points_fun[:,1].reshape((nx,ny))
z_surf = points_fun[:,2].reshape((nx,ny))

# 3D Plot
fig = mpl_plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131,projection='3d')
ax2 = fig.add_subplot(132,projection='3d')
ax3 = fig.add_subplot(133,projection='3d')

plotType = 'randn'
#plotType = 'lattice'
for ax in [ax1,ax2,ax3]:
    # Surface
    ax.plot_surface(x_surf,y_surf,z_surf,cmap='winter',alpha=.2)
    # Scatters
    points_distrib = zeros((n,dim+1))
    if plotType == 'lattice':
        muhat = zeros(j)
        set_x = get_RS_lattice_b2(n,dim,j) 
        for i,xu in enumerate(set_x):
            points_distrib[:,:2] = sqrt(var)*norm.ppf(xu) # transform
            points_distrib[:,2] = fun.g(points_distrib[:,:2],coordIdx)
            ax.scatter(points_distrib[:,0],points_distrib[:,1],points_distrib[:,2],color=colors[i],s=15)
            muhat[i] = points_distrib[:,2].mean()
        mu = muhat.mean()
        std2 = muhat.std()
        ax.set_title('\t$n$ = %-5d $\sigma$ = %-10.4f $\hat{\mu}$ = %-10.4f'%(n,std2,mu),fontdict={'fontsize': 14})
    if plotType == 'randn':
        xu = random.randn(n,dim)
        points_distrib[:,:2] = xu*sqrt(1/2) # transform
        points_distrib[:,2] = fun.g(points_distrib[:,:2],coordIdx)
        ax.scatter(points_distrib[:,0],points_distrib[:,1],points_distrib[:,2],color='g',s=5)
        mu = points_distrib[:,2].mean()
        std2 = points_distrib[:,2].std()
        epsilon = -norm.ppf(.01/2) * 1.2 * (std2**2/n).sum(0)**.5
        ax.set_title('\t$n$ = %-5d $\epsilon$ = %-10.4f $\hat{\mu}$ = %-10.4f'%(n,epsilon,mu),fontdict={'fontsize': 14})
    # axis metas
    n *= 2
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(20,45)
    
# Output
mpl_plt.savefig('DevelopOnly/Outputs/Three_3d_SurfaceScatters.png',
        dpi = 500,
        bbox_inches = 'tight',
        pad_inches = .05)
mpl_plt.show()
