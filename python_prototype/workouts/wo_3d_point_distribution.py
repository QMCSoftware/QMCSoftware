#!/usr/bin/python_prototype
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as mpl_plt
from numpy import arange, linspace, meshgrid, random, zeros
from scipy.stats import norm

from qmcpy._util import summarize
from qmcpy.distribution import IIDDistribution, QuasiRandom
from qmcpy.integrand import Keister
from qmcpy.measures import StdGaussian, Lattice, IIDZeroMeanGaussian

def plot3d(verbose=True):
    random.seed(7)
    dim = 2
    j = 3
    colors = ["r", "b", "g"]
    n = 32
    var = 1 / 2
    coordIdx = arange(1, dim + 1)

    fun = Keister()
    measure = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
    distribution = IIDDistribution(true_distribution=StdGaussian(dimension=[dim]), seed_rng=7)
    fun.transform_variable(measure, distribution)

    # Examples for generating "pregen" figure"s constants
    '''
    import sys
    from python_prototype.stop.clt_rep import CLTRep
    from python_prototype.stop.clt_stopping import CLT
    from python_prototype.integrate import integrate
    #     CLTRep Example
    funObj = Keister()
    distribution = QuasiRandom(true_distribution=Lattice(dimension=[dim]),seed_rng=7)
    stopObj = CLTRep(distribution,n_init=n,n_max=2**15,abs_tol=.01)
    measure = IIDZeroMeanGaussian(dimension=[dim],variance=[var])
    sol,dataObj = integrate(funObj,measure,distribution,stopObj)
    summarize(stopObj,measure,funObj,distribution,dataObj)
    #     CLT Example
    funObj = Keister()
    distribution = IIDDistribution(true_distribution=StdGaussian(dimension=[dim]),seed_rng=7)
    stopObj = CLT(distribution,n_init=16,abs_tol=.3,alpha=.01,inflate=1.2)
    measure = IIDZeroMeanGaussian(dimension=[dim],variance=[1/2])
    sol,dataObj = integrate(funObj,measure,distribution,stopObj)
    summarize(stopObj,measure,funObj,distribution,dataObj)
    sys.exit(0)
    '''

    # "pregen" constants based on running the above CLT Example
    eps_list = [.5, .4, .3]
    n_list = [50, 68, 109]
    muHat_list = [1.876, 1.806, 1.883]

    # Function Points

    nx, ny = (100, 100)
    points_fun = zeros((nx * ny, 3))
    x = linspace(-3, 3, nx)
    y = linspace(-3, 3, ny)
    x_2d, y_2d = meshgrid(x, y)
    points_fun[:, 0] = x_2d.flatten()
    points_fun[:, 1] = y_2d.flatten()
    points_fun[:, 2] = fun.f(points_fun[:, :2], coordIdx)
    x_surf = points_fun[:, 0].reshape((nx, ny))
    y_surf = points_fun[:, 1].reshape((nx, ny))
    z_surf = points_fun[:, 2].reshape((nx, ny))

    # 3D Plot
    fig = mpl_plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    plotType = "pregen"  # "randn" "lattice"
    # plotType = "lattice"
    for idx, ax in enumerate([ax1, ax2, ax3]):
        # Surface
        ax.plot_surface(x_surf, y_surf, z_surf, cmap="winter", alpha=.2)
        # Scatters
        points = zeros((n, dim + 1))
        if plotType == "lattice":
            muhat = zeros(j)
            set_x = QuasiRandom().get_RS_lattice_b2(n, dim, j)
            for i, xu in enumerate(set_x):
                points[:, :2] = xu
                points[:, 2] = fun.f(points[:, :2], coordIdx)
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           color=colors[i], s=15)
                muhat[i] = points[:, 2].mean()
            mu = muhat.mean()
            epsilon = muhat.std()
        if plotType == "randn":
            points[:, :2] = random.randn(n, dim)
            points[:, 2] = fun.f(points[:, :2], coordIdx)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="r", s=5)
            mu = points[:, 2].mean()
            std2 = points[:, 2].std()
            epsilon = -norm.ppf(.01 / 2) * 1.2 * (std2 ** 2 / n).sum(0) ** .5
        if plotType == "pregen":
            points[:, :2] = random.randn(n, dim)
            points[:, 2] = fun.f(points[:, :2], coordIdx)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="r", s=5)
            n = n_list[idx]
            epsilon = eps_list[idx]
            mu = muHat_list[idx]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="r", s=5)
        ax.set_title("\t$\epsilon$ = %-7.1f $n$ = %-7d $\hat{\mu}$ = %-7.2f " \
                     % (epsilon, n, mu), fontdict={"fontsize": 14})
        # axis metas
        n *= 2
        ax.grid(False)
        ax.xaxis.pane.set_edgecolor("black")
        ax.yaxis.pane.set_edgecolor("black")
        ax.set_xlabel("$x_1$", fontdict={"fontsize": 14})
        ax.set_ylabel("$x_2$", fontdict={"fontsize": 14})
        ax.set_zlabel("$f\:(x_1,x_2)$", fontdict={"fontsize": 14})
        ax.view_init(20, 45)

    # Output
    mpl_plt.savefig("outputs/Three_3d_SurfaceScatters.png",
                    dpi=500, bbox_inches="tight", pad_inches=.15)
    mpl_plt.show(block=False)

if __name__ == "__main__":
    plot3d()