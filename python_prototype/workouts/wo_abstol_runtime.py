#!/usr/bin/python_prototype/
from time import time
from matplotlib import pyplot as mpl_plot
from numpy import arange, array
import numpy as np
import pandas as pd
from copy import deepcopy

from qmcpy import integrate
from qmcpy._util import summarize
from qmcpy.integrand import AsianCall
from qmcpy.discrete_distribution import IIDStdGaussian,IIDStdUniform,Lattice,Sobol
from qmcpy.true_distribution import BrownianMotion
from qmcpy.stop import CLT, CLTRep

def plot(title,xlabel,ylabel,xdata,ydata,outF):
    #mpl_plot.title(title)
    mpl_plot.xlabel(xlabel,fontsize=14)
    mpl_plot.ylabel(ylabel,fontsize=14)
    for name,(trend,color) in ydata.items():
        mpl_plot.loglog(xdata,trend,basex=10,basey=10,color=color,label=name)
    #mpl_plot.xticks([5*10**-2,10**-1],fontsize=12)
    #mpl_plot.yticks([0,10,20,30],fontsize=12)
    mpl_plot.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01),
                    ncol=2, borderaxespad=0, frameon=False, prop={"size": 14})
    mpl_plot.savefig(outF+".png",
        dpi=500,
        bbox_inches = "tight",
        pad_inches = .05)
    mpl_plot.show(block=False)

# Constants
time_vector=[
    arange(1 / 4, 5 / 4, 1 / 4),
    arange(1 / 16, 17 / 16, 1 / 16),
    arange(1 / 64, 65 / 64, 1 / 64)]
dims = [len(tv) for tv in time_vector]

def QMC_Wrapper(discrete_distrib, true_distrib, stop, name):
    item_f = "    %-25s %-10.3f %-10.3f"
    item_s = "    %-25s %-10s %-10s"
    option = AsianCall(true_distrib)
    sol, data = integrate(option, discrete_distrib, true_distrib, stop)
    print(item_f % (name, sol, data.t_total))
    return sol, data.t_total

def comp_Clt_vs_cltRep_runtimes(abstols):
    df_metrics = pd.DataFrame({"abs_tol":[],
        "CLT_IIDStdUniform_sol":[],     "CLT_IIDStdUniform_runTime":[],
        "CLT_IIDStdGaussian_sol":[],    "CLT_IIDStdGaussian_runTime":[],
        "CLT_Rep_Lattice_sol":[],    "CLT_Rep_Lattice_runTime":[],
        "CLT_Rep_Sobol_sol":[],      "CLT_Rep_Sobol_runTime":[]})
    for i, abs_tol in enumerate(abstols):
        print("abs_tol: %-10.3f" % abs_tol)
        results = []  # hold row of DataFrame
        results.append(abs_tol)

        # CLT_IIDStdUniform
        discrete_distrib = IIDStdUniform()
        true_distrib = BrownianMotion(dims,time_vector=time_vector)
        stop = CLT(discrete_distrib, true_distrib, abs_tol=abs_tol)
        mu, t = QMC_Wrapper(discrete_distrib, true_distrib, stop, "CLT_IIDStdUniform")
        results.extend([mu, t])

        # CLT_IIDStdGaussian
        discrete_distrib = IIDStdGaussian()
        true_distrib = BrownianMotion(dims,time_vector=time_vector)
        stop = CLT(discrete_distrib, true_distrib, abs_tol=abs_tol)
        mu, t = QMC_Wrapper(discrete_distrib, true_distrib, stop, "CLT_IIDStdGaussian")
        results.extend([mu, t])

        # CLT_Rep_Lattice
        discrete_distrib = Lattice()
        true_distrib = BrownianMotion(dims,time_vector=time_vector)
        stop = CLTRep(discrete_distrib, true_distrib, abs_tol=abs_tol)
        mu, t = QMC_Wrapper(discrete_distrib, true_distrib, stop,  "CLT_Rep_Lattice")
        results.extend([mu, t])

        # CLT_Rep_Sobol
        discrete_distrib = Sobol()
        true_distrib = BrownianMotion(dims,time_vector=time_vector)
        stop = CLTRep(discrete_distrib, true_distrib, abs_tol=abs_tol)
        mu, t = QMC_Wrapper(discrete_distrib, true_distrib, stop, "CLT_Rep_Sobol")
        results.extend([mu, t])

        df_metrics.loc[i] = results # update metrics
    return df_metrics

if __name__ == "__main__":
    outF = "outputs/Compare_true_distribution_and_StoppingCriterion_vs_Abstol"
    # Run Test

    absTols = arange(.01, .051, .002)  # arange(.01,.06,.01)
    df_metrics = comp_Clt_vs_cltRep_runtimes(absTols)
    df_metrics.to_csv(outF + ".csv", index=False)

    # Gen Plot
    df = pd.read_csv(outF + ".csv")
    plot(title="Integration Time by Absolute Tolerance"
               + "\nfor Multi-level Asian Option Function",
         xlabel="Absolute Tolerance", ylabel="Integration Runtime",
         xdata=df["abs_tol"].values,
         ydata={"CLT: IID Gaussian": (df["CLT_IIDStdUniform_runTime"], "r"),
                "CLT: IID Uniform ": (df["CLT_IIDStdGaussian_runTime"], "b"),
                "CLT Repeated: Lattice": (df["CLT_Rep_Lattice_runTime"], "g"),
                "CLT Repeated: sobol": (df["CLT_Rep_Sobol_runTime"], "y")},
         outF=outF)
