from qmcpy import *
from numpy import *
import matplotlib 
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
from matplotlib import pyplot
fs = 20
pyplot.rc('font', size=fs)
pyplot.rc('axes', titlesize=fs)
pyplot.rc('axes', labelsize=fs)
pyplot.rc('xtick', labelsize=fs)
pyplot.rc('ytick', labelsize=fs)
pyplot.rc('legend', fontsize=fs)
pyplot.rc('figure', titlesize=fs)

# parameterss
n_mesh = 1002
n_ld = 2**7
d = 2
# qmcpy objects
dd = Sobol(d,seed=7)
m = Gaussian(dd,covariance=1./2)
k = Keister(m)
#   top surface
mesht = zeros(((n_mesh)**2,3),dtype=float)
grid_tics = linspace(-2,2,n_mesh)
x_mesh_t,y_mesh_t = meshgrid(grid_tics,grid_tics)
mesht[:,0] = x_mesh_t.flatten()
mesht[:,1] = y_mesh_t.flatten()
mesht[:,2] = k.g(mesht[:,:2]).squeeze()
z_mesh_t = mesht[:,2].reshape((n_mesh,n_mesh))
#   top points
ptst = zeros((n_ld,2),dtype=float)
ptst[:,:2] = m.gen_mimic_samples(n_ld)
#   bottom surface
meshb = zeros(((n_mesh-2)**2,3),dtype=float)
grid_tics = linspace(0,1,n_mesh)[1:-1]
x_mesh_b,y_mesh_b = meshgrid(grid_tics,grid_tics)
meshb[:,0] = x_mesh_b.flatten()
meshb[:,1] = y_mesh_b.flatten()
meshb[:,2] = k.f(meshb[:,:2]).squeeze()
z_mesh_b = meshb[:,2].reshape((n_mesh-2,n_mesh-2))
#   bottom points
ptsb = zeros((n_ld,2),dtype=float)
ptsb[:,:2] = dd.gen_samples(n_ld)
# plots
fig,ax = pyplot.subplots(figsize=(10,5),nrows=1,ncols=2)
#   colors 
z_min = min(mesht[:,2].min(),mesht[:,2].min())
z_max = max(meshb[:,2].max(),meshb[:,2].max())
clevel = arange(z_min,z_max,.025)
#cmap = pyplot.get_cmap('Blues')
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(.95,.95,.95),(0,0,1)])
#   contours
ax[0].contourf(x_mesh_t,y_mesh_t,z_mesh_t,clevel,cmap=cmap,extend='both')
ax[1].contourf(x_mesh_b,y_mesh_b,z_mesh_b,clevel,cmap=cmap,extend='both')
#   scatters
ax[0].scatter(ptst[:,0],ptst[:,1],s=5,color='w')
ax[1].scatter(ptsb[:,0],ptsb[:,1],s=5,color='w')
#   axis
lims = [[-2,2],[0,1]]
for i in range(2):
    for nsew in ['top','bottom','left','right']: ax[i].spines[nsew].set_visible(False)
    ax[i].xaxis.set_ticks_position('none') 
    ax[i].yaxis.set_ticks_position('none') 
    lim = lims[i]
    ax[i].set_aspect(1)
    ax[i].set_xlim(lim)
    ax[i].set_xticks(lim)
    ax[i].set_ylim(lim)
    ax[i].set_yticks(lim)
#   labels
ax[0].set_xlabel('$t_1$')
ax[0].set_ylabel('$t_2$')
ax[0].set_title(r'$g(\boldsymbol{t})$')
ax[1].set_xlabel('$x_1$')
ax[1].set_ylabel('$x_1$')
ax[1].set_title(r'$f(\boldsymbol{x})$')
#   metas
fig.tight_layout()
pyplot.savefig("presentations_papers/sorokin_masters_thesis/figs/i_keister_contours.png",
    dpi=100,transparent=True)
pyplot.show()