""" Integrations problems to vary parameters on """

from qmcpy import *

def cubmcclt_iidstduniform(dimension, abs_tol, drift):
    distribution = IIDStdUniform(dimension,seed=7)
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianCall(measure)
    solution,data = CubMCCLT(integrand, abs_tol).integrate()
    return data

def cubmcclt_iidstdgaussian(dimension, abs_tol, drift):
    distribution = IIDStdGaussian(dimension,seed=7)
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianCall(measure)
    solution,data = CubMCCLT(integrand, abs_tol).integrate()
    return data

def cubmcg_iidstduniform(dimension, abs_tol, drift):
    distribution = IIDStdUniform(dimension,seed=7)
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianCall(measure)
    solution,data = CubMCG(integrand, abs_tol).integrate()
    return data

def cubmcg_iidstdgaussian(dimension, abs_tol, drift):
    distribution = IIDStdGaussian(dimension,seed=7)
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianCall(measure)
    solution,data = CubMCG(integrand, abs_tol).integrate()
    return data

def cubqmcclt_lattice(dimension, abs_tol, drift):
    distribution = Lattice(dimension, seed=7, backend="MPS")
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianCall(measure)
    solution,data = CubQMCCLT(integrand, abs_tol).integrate()
    return data

def cubqmcclt_sobol(dimension, abs_tol, drift):
    distribution = Sobol(dimension, seed=7, backend="QRNG")
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianCall(measure)
    solution,data = CubQMCCLT(integrand, abs_tol).integrate()
    return data

def cubqmclatticeg(dimension, abs_tol, drift):
    distribution = Lattice(dimension, seed=7, backend="GAIL")
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianCall(measure)
    solution,data = CubQMCLatticeG(integrand, abs_tol,).integrate()
    return data

def cubqmcsobolg(dimension, abs_tol, drift):
    distribution = Sobol(dimension, seed=7, backend="QRNG")
    measure = BrownianMotion(distribution, drift=drift)
    integrand = AsianCall(measure)
    solution,data = CubQMCSobolG(integrand, abs_tol).integrate()
    return data

integrations_dict = {
    ('CubMCCLT','IIDStdUniform','MC'): cubmcclt_iidstduniform,
    ('CubMCCLT','IIDStdGaussian','MC'): cubmcclt_iidstdgaussian,
    ('CubMCG','IIDStdUniform','MC'): cubmcg_iidstduniform,
    ('CubMCG','IIDStdGaussian','MC'): cubmcg_iidstdgaussian,
    ('CubQMCCLT','Lattice','QMC'): cubqmcclt_lattice,
    ('CubQMCCLT','Sobol','QMC'): cubqmcclt_sobol,
    ('CubQMCLatticeG','Lattice','QMC'): cubqmclatticeg,
    ('CubQMCSobolG','Sobol','QMC'): cubqmcsobolg}