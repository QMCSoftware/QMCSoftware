""" Integrations problems to vary parameters on """

from qmcpy import *

def cubmcclt_iidstduniform(dimension, abs_tol, drift):
    integrand = AsianOption(BrownianMotion(IIDStdUniform(dimension,seed=7),drift=drift))
    solution,data = CubMCCLT(integrand, abs_tol).integrate()
    return data

def cubmcg_iidstduniform(dimension, abs_tol, drift):
    integrand = AsianOption(BrownianMotion(IIDStdUniform(dimension,seed=7),drift=drift))
    solution,data = CubMCG(integrand, abs_tol).integrate()
    return data

def cubqmcclt_lattice(dimension, abs_tol, drift):
    integrand = AsianOption(BrownianMotion(Lattice(dimension,seed=7),drift=drift))
    solution,data = CubQMCCLT(integrand, abs_tol).integrate()
    return data

def cubqmcclt_sobol(dimension, abs_tol, drift):
    integrand = AsianOption(BrownianMotion(Sobol(dimension,seed=7),drift=drift))
    solution,data = CubQMCCLT(integrand, abs_tol).integrate()
    return data

def cubqmclatticeg(dimension, abs_tol, drift):
    integrand = AsianOption(BrownianMotion(Lattice(dimension,seed=7),drift=drift))
    solution,data = CubQMCLatticeG(integrand, abs_tol).integrate()
    return data

def cubqmcsobolg(dimension, abs_tol, drift):
    integrand = AsianOption(BrownianMotion(Sobol(dimension,seed=7),drift=drift))
    solution,data = CubQMCSobolG(integrand, abs_tol).integrate()
    return data

def cubbayeslatticeg(dimension, abs_tol, drift):
    integrand = AsianOption(BrownianMotion(Lattice(dimension,seed=7,order='linear'),drift=drift))
    solution,data = CubBayesLatticeG(integrand, abs_tol).integrate()
    return data

def cubbayesnetg(dimension, abs_tol, drift):
    integrand = AsianOption(BrownianMotion(Sobol(dimension,seed=7),drift=drift))
    solution,data = CubBayesNetG(integrand, abs_tol).integrate()
    return data

integrations_dict = {
    ('CubMCCLT','IIDStdUniform','MC'): cubmcclt_iidstduniform,
    ('CubMCG','IIDStdUniform','MC'): cubmcg_iidstduniform,
    ('CubQMCCLT','Lattice','QMC'): cubqmcclt_lattice,
    ('CubQMCCLT','Sobol','QMC'): cubqmcclt_sobol,
    ('CubQMCLatticeG','Lattice','QMC'): cubqmclatticeg,
    ('CubQMCSobolG','Sobol','QMC'): cubqmcsobolg}
    #('CubBayesLatticeG','Lattice','QMC'): cubbayeslatticeg,
    #('CubBayesNetG','Sobol','QMC'): cubbayesnetg}