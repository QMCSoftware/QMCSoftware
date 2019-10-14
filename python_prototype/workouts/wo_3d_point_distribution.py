#!/usr/bin/python_prototype
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as mpl_plt
from numpy import linspace, meshgrid, random, zeros, sqrt

from qmcpy.integrand import Keister


def plot3d(verbose=True):
    # Compute n_total and mu_hat for each epsilon with a cooresponding plot
    '''
    integrand = Keister()
    discrete_distrib = IIDStdGaussian()
    true_measure = Gaussian(dimension=2,variance=1/2)
    stop = CLT(discrete_distrib,true_measure, abs_tol=.5, n_init=16, n_max=1e10)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()
    sys.exit(0)
    '''

    # Constants based on running the above CLT Example
    eps_list = [.5, .4, .3]
    n_list = [50, 68, 109]
    muHat_list = [1.876, 1.806, 1.883]

    fun = Keister()
    colors = ["r", "b", "g"]
    n = 32

    # Function Points
    nx, ny = (100, 100)
    points_fun = zeros((nx * ny, 3))
    x = linspace(-3, 3, nx)
    y = linspace(-3, 3, ny)
    x_2d, y_2d = meshgrid(x, y)
    points_fun[:, 0] = x_2d.flatten()
    points_fun[:, 1] = y_2d.flatten()
    points_fun[:, 2] = fun.g(points_fun[:, :2])
    x_surf = points_fun[:, 0].reshape((nx, ny))
    y_surf = points_fun[:, 1].reshape((nx, ny))
    z_surf = points_fun[:, 2].reshape((nx, ny))

    # 3D Plot
    fig = mpl_plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    for idx, ax in enumerate([ax1, ax2, ax3]):
        # Surface
        ax.plot_surface(x_surf, y_surf, z_surf, cmap="winter", alpha=.2)
        # Scatters
        points = zeros((n, 3))
        points[:, :2] = random.randn(n, 2)*sqrt(1/2)
        points[:, 2] = fun.g(points[:, :2])
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="r", s=5)
        n = n_list[idx]
        epsilon = eps_list[idx]
        mu = muHat_list[idx]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="r", s=5)
        ax.set_title("\t$\epsilon$ = %-7.1f $n$ = %-7d $\hat{\mu}$ = %-7.2f "
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
