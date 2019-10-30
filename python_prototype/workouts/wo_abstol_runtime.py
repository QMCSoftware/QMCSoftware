"""
Plot run time vs. absolute tolerances for multi-level Asian option function.
"""
from matplotlib import pyplot as mpl_plot
from numpy import arange,nan
import pandas as pd

from qmcpy import integrate
from qmcpy.discrete_distribution import IIDStdGaussian, IIDStdUniform, Lattice, Sobol
from qmcpy.integrand import AsianCall
from qmcpy.stopping_criterion import CLT, CLTRep
from qmcpy.true_measure import BrownianMotion


def plot(title, xlabel, ylabel, xdata, ydata, outF):
    """
    Plot run time against abolute tolerances.

    Args:
        title (str): plot title.
        xlabel (str): label for x-axis.
        ylabel (str): label for y-axis
        xdata (list): list of absolute tolerances.
        ydata (list): list of run time.
        outF (str): location of image .png file.

    Returns:
        None

    """
    mpl_plot.cla()  # Clear axis
    mpl_plot.clf()  # Clear figure
    mpl_plot.title(title)
    mpl_plot.xlabel(xlabel, fontsize=14)
    mpl_plot.ylabel(ylabel, fontsize=14)
    for name, (trend, color) in ydata.items():
        mpl_plot.loglog(xdata, trend, basex=10, basey=10, color=color, label=name)
    # mpl_plot.xticks([5*10**-2,10**-1],fontsize=12)
    # mpl_plot.yticks([0,10,20,30],fontsize=12)
    mpl_plot.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
                    borderaxespad=0, frameon=False, prop={"size": 14})
    mpl_plot.savefig(outF + ".png", dpi=500, bbox_inches="tight", pad_inches=.05)
    mpl_plot.show(block=False)


# Constants
time_vector = [
    arange(1 / 4, 5 / 4, 1 / 4),
    arange(1 / 16, 17 / 16, 1 / 16),
    arange(1 / 64, 65 / 64, 1 / 64)]
dims = [len(tv) for tv in time_vector]


def qmc_wrapper(discrete_distrib, true_measure, stopping_criterion, name):
    """
    Call QMCPy's integrate function given inputs

    Args:
        discrete_distrib: instance of discrete distribution.
        true_measure: true measure for Asian pption.
        stopping_criterion: stopping criterion for QMCPy's integrate algorithm.
        name: label for stopping criterion.

    Returns:
        sol: solution estimate
        time_total: total run time for obtaining solution estimate

    """
    item_f = "    %-25s %-10.3f %-10.3f"
    option = AsianCall(true_measure)
    try:
        sol, data = integrate(option, true_measure, discrete_distrib, stopping_criterion)
        print(item_f % (name, sol, data.time_total))
        return sol, data.time_total
    except:
        return nan,nan


def comp_clt_vs_cltrep_runtimes(abstols):
    """
    Collect run time statistics, given absolute tolerances.

    Args:
        abstols: list of absolute tolerances

    Returns:
        df_metrics (pandas.DataFrame):

    """
    df_metrics = pd.DataFrame({"abs_tol": [],
                               "CLT_IIDStdUniform_sol": [], "CLT_IIDStdUniform_runTime": [],
                               "CLT_IIDStdGaussian_sol": [], "CLT_IIDStdGaussian_runTime": [],
                               "CLT_Rep_Lattice_sol": [], "CLT_Rep_Lattice_runTime": [],
                               "CLT_Rep_Sobol_sol": [], "CLT_Rep_Sobol_runTime": []})
    for i, abs_tol in enumerate(abstols):
        print("abs_tol: %-10.3f" % abs_tol)
        results = []  # hold row of DataFrame
        results.append(abs_tol)

        # CLT_IIDStdUniform
        discrete_distrib = IIDStdUniform(rng_seed=7)
        true_measure = BrownianMotion(dims, time_vector=time_vector)
        stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=abs_tol, n_max=10e15)
        mu, t = qmc_wrapper(discrete_distrib, true_measure,
                            stopping_criterion, "CLT_IIDStdUniform")
        results.extend([mu, t])

        # CLT_IIDStdGaussian
        discrete_distrib = IIDStdGaussian(rng_seed=7)
        true_measure = BrownianMotion(dims, time_vector=time_vector)
        stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=abs_tol, n_max=10e15)
        mu, t = qmc_wrapper(discrete_distrib, true_measure,
                            stopping_criterion, "CLT_IIDStdGaussian")
        results.extend([mu, t])

        # CLT_Rep_Lattice
        discrete_distrib = Lattice(rng_seed=7)
        true_measure = BrownianMotion(dims, time_vector=time_vector)
        stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol, n_max=10e15)
        mu, t = qmc_wrapper(discrete_distrib, true_measure,
                            stopping_criterion, "CLT_Rep_Lattice")
        results.extend([mu, t])

        # CLT_Rep_Sobol
        discrete_distrib = Sobol(rng_seed=7)
        true_measure = BrownianMotion(dims, time_vector=time_vector)
        stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol, n_max=10e15)
        mu, t = qmc_wrapper(discrete_distrib, true_measure,
                            stopping_criterion, "CLT_Rep_Sobol")
        results.extend([mu, t])

        df_metrics.loc[i] = results  # update metrics

    return df_metrics


def plot_abstol_runtime(abstols=arange(.001, .021, .001), is_plot=True):
    """
    Integration Time by Absolute Tolerance for Multi-level Asian Option Function.

    Args:
        abstols (list): abolute tolerances.

    Returns:
        None
    """
    out_file = "outputs/Compare_true_distribution_and_StoppingCriterion_vs_Abstol"
    # Run Test
    df_metrics = comp_clt_vs_cltrep_runtimes(abstols)
    df_metrics.to_csv(out_file + ".csv", index=False)

    # Gen Plot
    df = pd.read_csv(out_file + ".csv")
    plot(title="",
         xlabel="Absolute Tolerance", ylabel="Integration Runtime",
         xdata=df["abs_tol"].values,
         ydata={"CLT: IID Gaussian": (df["CLT_IIDStdUniform_runTime"], "r"),
                "CLT: IID Uniform ": (df["CLT_IIDStdGaussian_runTime"], "b"),
                "CLT Repeated: Lattice": (df["CLT_Rep_Lattice_runTime"], "g"),
                "CLT Repeated: Sobol": (df["CLT_Rep_Sobol_runTime"], "y")},
         outF=out_file)


if __name__ == "__main__":
    plot_abstol_runtime(abstols=arange(.001, .1, .003))
