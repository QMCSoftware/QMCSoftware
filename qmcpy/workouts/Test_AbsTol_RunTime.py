from time import time

from algorithms.distribution import measure
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.function.AsianCallFun import AsianCallFun
from algorithms.integrate import integrate
from algorithms.stop.CLT_Rep import CLT_Rep
from algorithms.stop.CLTStopping import CLTStopping
from matplotlib import pyplot as mpl_plot
from numpy import arange, array
import numpy as np
import pandas as pd

def plot(title,xlabel,ylabel,xdata,ydata):
    #mpl_plot.title(title)
    mpl_plot.xlabel(xlabel,fontsize=14)
    mpl_plot.ylabel(ylabel,fontsize=14)
    for name,(trend,color) in ydata.items():
        mpl_plot.loglog(xdata,trend,basex=10,basey=10,color=color,label=name)
    #mpl_plot.xticks([5*10**-2,10**-1],fontsize=12)
    #mpl_plot.yticks([0,10,20,30],fontsize=12)
    mpl_plot.legend(
        loc = 'lower left',
        bbox_to_anchor = (0.0, 1.01),
        ncol = 2, 
        borderaxespad = 0,
        frameon = False,
        prop = {'size': 14})
    mpl_plot.savefig('workouts/Outputs/AbsTol_Runtime_LinePlot.png',
        dpi=500,
        bbox_inches = 'tight',
        pad_inches = .05)
    mpl_plot.show()

def QMC_Wrapper(stopObj,distribObj):
    measureObj = measure().BrownianMotion(
        timeVector=[arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)])
    OptionObj = AsianCallFun(measureObj)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    return sol,dataObj.timeUsed

def comp_Clt_vs_cltRep_runtimes(abstols):
    item_f = '    %-25s %-10.3f %-10.3f'
    item_s = '    %-25s %-10s %-10s'
    df_metrics = pd.DataFrame({'absTol':[],
        'CLT_stdUniform_sol':[],     'CLT_stdUniform_runTime':[],
        'CLT_stdGaussian_sol':[],    'CLT_stdGaussian_runTime':[],
        'CLT_Rep_lattice_sol':[],    'CLT_Rep_lattice_runTime':[],
        'CLT_Rep_Sobol_sol':[],      'CLT_Rep_Sobol_runTime':[]})
    for i,absTol in enumerate(abstols):        
        print('absTol: %-10.3f'%absTol)
        results = [] # hold row of DataFrame
        results.append(absTol)

        # CLT_stdUniform
        try:
            mu,t =\
            QMC_Wrapper(
                CLTStopping(absTol=absTol),
                IIDDistribution(trueD=measure().stdUniform(dimension=[4,16,64]),rngSeed=7))
            print(item_f%('CLT_stdUniform',mu,t))
        except:
            mu,t= '',''
            print(item_s%('CLT_stdUniform',mu,t))
        results.extend([mu,t])
        
        # CLT_stdGaussian
        try:
            mu,t =\
            QMC_Wrapper(
                 CLTStopping(absTol=absTol),
                 IIDDistribution(trueD=measure().stdGaussian(dimension=[4,16,64]),rngSeed=7))
            print(item_f%('CLT_stdGaussian',mu,t))
        except:
            mu,t = '',''
            print(item_s%('CLT_stdGaussian',mu,t))
        results.extend([mu,t])

        # CLT_Rep_lattice      
        try:
            mu,t =\
            QMC_Wrapper(
                CLT_Rep(nMax=2**20,absTol=absTol),
                QuasiRandom(trueD=measure().lattice(dimension=[4,16,64]),rngSeed=7))
            print(item_f%('CLT_Rep_lattice',mu,t))
        except:
            mu,t = '',''
            print(item_s%('CLT_Rep_lattice',mu,t))
        results.extend([mu,t])

        # CLT_Rep_sobol
        try:
            mu,t =\
            QMC_Wrapper(
                CLT_Rep(nMax=2**20,absTol=absTol),
                QuasiRandom(trueD=measure().Sobol(dimension=[4,16,64]),rngSeed=7))
            print(item_f%('CLT_Rep_sobol',mu,t))
        except:
            mu,t = '',''
            print(item_s%('CLT_Rep_lattice',mu,t))
        results.extend([mu,t])

        df_metrics.loc[i] = results # update metrics
    return df_metrics

if __name__ == '__main__':
    outF = 'workouts/Outputs/Compare_TrueD_and_StoppingCriterion_vs_Abstol.csv'
    # Run Test
    absTols = arange(.001,.051,.002) # [10 ** (-i / 5) for i in range(15, 7, -1)]
    df_metrics = comp_Clt_vs_cltRep_runtimes(absTols)
    df_metrics.to_csv(outF,index=False)

    # Gen Plot
    df = pd.read_csv(outF)
    plot(title = 'Integration Time by Absolute Tolerance \nfor Multi-level Asian Option Function',
        xlabel = 'Absolute Tolerance',
        ylabel = 'Integration Runtime',
        xdata = df['absTol'].values,
        ydata = {
            'CLT: IID Gaussian':(df['CLT_stdUniform_runTime'],'r'),
            'CLT: IID Uniform ':(df['CLT_stdGaussian_runTime'],'b'),
            'CLT Repeated: Lattice':(df['CLT_Rep_lattice_runTime'],'g'),
            'CLT Repeated: Sobol':(df['CLT_Rep_Sobol_runTime'],'y')})