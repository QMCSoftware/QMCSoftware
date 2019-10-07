""" Meta-data nd public utilites for qmcpy """

name = "qmcpy"
__version__ = 0.1

def print_summary(stopObj, measureObj, funObj, distribObj, dataObj):
    """
    print a summary of the qmc problem after execution
    
    Args:
        stopObj (StoppingCriterion): a Stopping Criterion object
        measureObj (Measure): a Measure object
        funObj (Integrand): an Integrand object
        dataObj (dataObj): a AccumData object
    """
    h1 = '%s (%s)\n'
    item_i = '%25s: %d\n'
    item_f = '%25s: %-15.4f\n'
    item_s = '%25s: %-15s\n'
    s = 'Solution: %-15.4f\n%s\n'%(dataObj.solution,'~'*50)

    s += h1%(type(funObj).__name__,'Function Object')

    s += h1%(type(measureObj).__name__,'Measure Type')
    s += item_s%('dimension',str(measureObj.dimension))

    s += h1%(type(distribObj).__name__,'Distribution Object')
    s += item_s%('true_distribution.measureName',type(distribObj.true_distribution).__name__)

    s += h1%(type(stopObj).__name__,'StoppingCriterion Object')
    s += item_f%('abs_tol',stopObj.abs_tol)
    s += item_f%('rel_tol',stopObj.rel_tol)
    s += item_i%('n_max',stopObj.n_max)
    s += item_f%('inflate',stopObj.inflate)
    s += item_f%('alpha',stopObj.alpha)

    s += h1%(type(dataObj).__name__,'Data Object')
    s += item_s%('n_samples_total',str(dataObj.n_samples_total))
    s += item_f%('t_total',dataObj.t_total)
    s += item_s%('confid_int',str(dataObj.confid_int))

    print(s)
    return

# API
from .integrate import integrate
from ._util import DistributionCompatibilityError, univ_repr
from .accum_data.mean_var_data import MeanVarData
from .accum_data.mean_var_data_rep import MeanVarDataRep
from .distribution.iid_distribution import IIDDistribution
from .distribution.quasi_random import QuasiRandom
from .integrand.asian_call import AsianCall
from .integrand.keister import Keister
from .integrand.linear import Linear
from .measures.measures import StdUniform,StdGaussian,IIDZeroMeanGaussian,BrownianMotion,Lattice,Sobol
from .stop.clt import CLT
from .stop.clt_rep import CLTRep
