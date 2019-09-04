from time import time
import pandas as pd
from matplotlib import pyplot as mpl_plot
from numpy import arange

from algorithms.function.AsianCallFun import AsianCallFun
from algorithms.stop.CLTStopping import CLTStopping
from algorithms.stop.CLT_Rep import CLT_Rep
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution.Mesh import Mesh
from algorithms.integrate import integrate
from algorithms.distribution import measure

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
    measureObj = measure().BrownianMotion(timeVector=[arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)])
    OptionObj = AsianCallFun(measureObj) # multi-level
    t0 = time()
    sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)
    t_delta = time()-t0
    return sol,t_delta

def comp_Clt_vs_cltRep_runtimes(fname,abstols):
    ''' Graph program-time by varying abstol '''
    # Open File and print headers
    f = open(fname,'w')
    f.write('%s,%s,%s,%s'%\
        ('CLT_stdUniform','CLT_stdGaussian','CLT_Rep_lattice','CLT_Rep_Sobol'))
    
    for absTol in abstols:
        print('Absolute Tolerance:',absTol)
        # CLT_stdUniform
        try: sol,tDelta = QMC_Wrapper(CLTStopping(absTol=absTol),IIDDistribution(trueD=measure().stdUniform(dimension=[4,16,64])))
        except: sol,tDelta = '',''
        f.write('\n'+str(tDelta)+',')
        print('\tCLT_stdUniform:',sol,tDelta)
        # CLT_stdGaussian
        try: sol,tDelta = QMC_Wrapper(CLTStopping(absTol=absTol),IIDDistribution(trueD=measure().stdGaussian(dimension=[4,16,64])))
        except: sol,tDelta = '',''
        f.write(str(tDelta)+',')
        print('\tCLT_stdGaussian:',sol,tDelta)
        # CLT_Rep_lattice
        try: sol,tDelta = QMC_Wrapper(CLT_Rep(nMax=2**20,absTol=absTol),Mesh(trueD=measure().mesh(dimension=[4,16,64],meshType='lattice')))
        except: sol,tDelta = '',''
        f.write(str(tDelta)+',')
        print('\tCLT_Rep_lattice:',sol,tDelta)
        # CLT_Rep_sobol (Commented out until sobol is improved)
        '''
        try: sol,tDelta = QMC_Wrapper(CLT_Rep(nMax=2**20,absTol=absTol),Mesh(trueD=measure().mesh(dimension=[4,16,64],meshType='sobol')))
        except: sol,tDelta = '',''
        f.write(str(tDelta))
        print('\tCLT_Rep_sobol:',sol,tDelta)
        '''
    f.close()  
    
if __name__ == '__main__':
    # Generate Times CSV
    fname = 'workouts/Outputs/Compare_TrueD_and_StoppingCriterion_vs_Abstol.csv'
    absTols = arange(.001,.021,.001)#arange(.001,.011,.001)
    #comp_Clt_vs_cltRep_runtimes(fname,absTols)
    
    df = pd.read_csv(fname)
    plot(title = 'Integration Time by Absolute Tolerance \nfor Multi-level Asian Option Function',
        xlabel = 'Absolute Tolerance',
        ylabel = 'Integration Runtime',
        xdata = absTols,
        ydata = {
            'CLT: IID Gaussian':(df.CLT_stdUniform,'r'),
            'CLT: IID Uniform ':(df.CLT_stdGaussian,'b'),
            'CLT Repeated: Lattice':(df.CLT_Rep_lattice,'g')})
            #'CLT Repeated: Sobol':(df.CLT_Rep_Sobol,'y')})