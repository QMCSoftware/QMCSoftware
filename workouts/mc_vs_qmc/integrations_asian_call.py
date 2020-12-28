""" Integrations problems to vary parameters on """

from qmcpy import *

def cubmcclt_iidstduniform(dimension, abs_tol, drift):
    distribution = IIDStdUniform(dimension,seed=7)
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianOption(measure)
    solution,data = CubMCCLT(integrand, abs_tol).integrate()
    return data

def cubmcg_iidstduniform(dimension, abs_tol, drift):
    distribution = IIDStdUniform(dimension,seed=7)
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianOption(measure)
    solution,data = CubMCG(integrand, abs_tol).integrate()
    return data

def cubqmcclt_lattice(dimension, abs_tol, drift):
    distribution = Lattice(dimension, seed=7, order='mps')
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianOption(measure)
    solution,data = CubQMCCLT(integrand, abs_tol).integrate()
    return data

def cubqmcclt_sobol(dimension, abs_tol, drift):
    distribution = Sobol(dimension, seed=7)
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianOption(measure)
    solution,data = CubQMCCLT(integrand, abs_tol).integrate()
    return data

def cubqmclatticeg(dimension, abs_tol, drift):
    distribution = Lattice(dimension, seed=7)
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianOption(measure)
    solution,data = CubQMCLatticeG(integrand, abs_tol,).integrate()
    return data

def cubbayeslatticeg(dimension, abs_tol, drift):
    distribution = Lattice(dimension, seed=7, order='linear', randomize=True)
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianOption(measure)
    solution,data = CubBayesLatticeG(integrand, abs_tol,).integrate()
    return data

def cubqmcsobolg(dimension, abs_tol, drift):
    distribution = Sobol(dimension, seed=7)
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianOption(measure)
    solution,data = CubQMCSobolG(integrand, abs_tol).integrate()
    return data

integrations_dict = {
    ('CubMCCLT','IIDStdUniform','MC'): cubmcclt_iidstduniform,
    ('CubMCG','IIDStdUniform','MC'): cubmcg_iidstduniform,
    ('CubQMCCLT','Lattice','QMC'): cubqmcclt_lattice,
    ('CubQMCCLT','Sobol','QMC'): cubqmcclt_sobol,
    ('CubQMCLatticeG','Lattice','QMC'): cubqmclatticeg,
    ('CubBayesLatticeG','Lattice','QMC'): cubbayeslatticeg,
    ('CubQMCSobolG','Sobol','QMC'): cubqmcsobolg}