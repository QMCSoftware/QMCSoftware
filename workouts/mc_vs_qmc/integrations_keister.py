""" Integrations problems to vary parameters on """

from qmcpy import *

def cubmcclt_iidstduniform(dimension, abs_tol, rel_tol):
    distribution = IIDStdUniform(dimension,seed=7)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubMCCLT(integrand, abs_tol, rel_tol).integrate()
    return data

def cubmcclt_iidstdgaussian(dimension, abs_tol, rel_tol):
    distribution = IIDStdGaussian(dimension,seed=7)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubMCCLT(integrand, abs_tol, rel_tol).integrate()
    return data

def cubmcg_iidstduniform(dimension, abs_tol, rel_tol):
    distribution = IIDStdUniform(dimension,seed=7)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubMCG(integrand, abs_tol, rel_tol).integrate()
    return data

def cubmcg_iidstdgaussian(dimension, abs_tol, rel_tol):
    distribution = IIDStdGaussian(dimension,seed=7)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubMCG(integrand, abs_tol, rel_tol).integrate()
    return data

def cubqmcclt_lattice(dimension, abs_tol, rel_tol):
    distribution = Lattice(dimension, randomize=True, seed=7, backend="MPS")
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubQMCCLT(integrand, abs_tol, rel_tol).integrate()
    return data

def cubqmcclt_sobol(dimension, abs_tol, rel_tol):
    distribution = Sobol(dimension, randomize=True, seed=7)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubQMCCLT(integrand, abs_tol, rel_tol).integrate()
    return data

def cubqmclatticeg(dimension, abs_tol, rel_tol):
    distribution = Lattice(dimension, seed=7, backend="GAIL")
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubQMCLatticeG(integrand, abs_tol, rel_tol).integrate()
    return data

def cubqmcsobolg(dimension, abs_tol, rel_tol):
    distribution = Sobol(dimension, seed=7)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubQMCSobolG(integrand, abs_tol, rel_tol).integrate()
    return data

def cubbayeslatticeg(dimension, abs_tol, rel_tol):
    distribution = Lattice(dimension, linear=True, backend="GAIL")
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution, data = CubBayesLatticeG(integrand, abs_tol, rel_tol).integrate()
    return data


integrations_dict = {
    ('CubMCCLT','IIDStdUniform','MC'): cubmcclt_iidstduniform,
    ('CubMCCLT','IIDStdGaussian','MC'): cubmcclt_iidstdgaussian,
    ('CubMCG','IIDStdUniform','MC'): cubmcg_iidstduniform,
    ('CubMCG','IIDStdGaussian','MC'): cubmcg_iidstdgaussian,
    ('CubQMCCLT','Lattice','QMC'): cubqmcclt_lattice,
    ('CubQMCCLT','Sobol','QMC'): cubqmcclt_sobol,
    ('CubQMCLatticeG','Lattice','QMC'): cubqmclatticeg,
    ('CubQMCSobolG','Sobol','QMC'): cubqmcsobolg,
    ('CubBayesLatticeG', 'Lattice', 'QMC'): cubbayeslatticeg,
}
