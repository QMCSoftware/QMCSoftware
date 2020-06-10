""" Integrations problems to vary parameters on """

from qmcpy import *

def cubmcclt_iidstduniform(dimension, abs_tol, rel_tol):
    distribution = IIDStdUniform(dimension,seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubMcClt(integrand, abs_tol, rel_tol).integrate()
    return data

def cubmcclt_iidstdgaussian(dimension, abs_tol, rel_tol):
    distribution = IIDStdGaussian(dimension,seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubMcClt(integrand, abs_tol, rel_tol).integrate()
    return data

def cubmcg_iidstduniform(dimension, abs_tol, rel_tol):
    distribution = IIDStdUniform(dimension,seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubMcG(integrand, abs_tol, rel_tol).integrate()
    return data

def cubmcg_iidstdgaussian(dimension, abs_tol, rel_tol):
    distribution = IIDStdGaussian(dimension,seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubMcG(integrand, abs_tol, rel_tol).integrate()
    return data

def cubqmcclt_lattice(dimension, abs_tol, rel_tol):
    distribution = Lattice(dimension, scramble=True, seed=7, backend="MPS")
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubQmcClt(integrand, abs_tol, rel_tol).integrate()
    return data

def cubqmcclt_sobol(dimension, abs_tol, rel_tol):
    distribution = Sobol(dimension, scramble=True, seed=7, backend="QRNG")
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubQmcClt(integrand, abs_tol, rel_tol).integrate()
    return data

def cubqmclatticeg(dimension, abs_tol, rel_tol):
    distribution = Lattice(dimension, seed=7, backend="GAIL")
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubQmcLatticeG(integrand, abs_tol, rel_tol).integrate()
    return data

def cubqmcsobolg(dimension, abs_tol, rel_tol):
    distribution = Sobol(dimension, seed=7, backend="QRNG")
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubQmcSobolG(integrand, abs_tol, rel_tol).integrate()
    return data

integrations_dict = {
    ('CubMcClt','IIDStdUniform','MC'): cubmcclt_iidstduniform,
    ('CubMcClt','IIDStdGaussian','MC'): cubmcclt_iidstdgaussian,
    ('CubMcG','IIDStdUniform','MC'): cubmcg_iidstduniform,
    ('CubMcG','IIDStdGaussian','MC'): cubmcg_iidstdgaussian,
    ('CubQmcClt','Lattice','QMC'): cubqmcclt_lattice,
    ('CubQmcClt','Sobol','QMC'): cubqmcclt_sobol,
    ('CubQmcLatticeG','Lattice','QMC'): cubqmclatticeg,
    ('CubQmcSobolG','Sobol','QMC'): cubqmcsobolg}