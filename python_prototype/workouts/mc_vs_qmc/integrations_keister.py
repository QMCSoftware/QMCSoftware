""" Integrations problems to vary parameters on """

from qmcpy import *

def clt_iidstduniform(dimension, abs_tol, rel_tol):
    distribution = IIDStdUniform(dimension,seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CLT(integrand, abs_tol, rel_tol).integrate()
    return data

def clt_iidstdgaussian(dimension, abs_tol, rel_tol):
    distribution = IIDStdGaussian(dimension,seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CLT(integrand, abs_tol, rel_tol).integrate()
    return data

def cltrep_lattice(dimension, abs_tol, rel_tol):
    distribution = Lattice(dimension, scramble=True, replications=16, seed=7, backend="MPS")
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CLTRep(integrand, abs_tol, rel_tol).integrate()
    return data

def cltrep_sobol(dimension, abs_tol, rel_tol):
    distribution = Sobol(dimension, scramble=True, replications=16, seed=7, backend="MPS")
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CLTRep(integrand, abs_tol, rel_tol).integrate()
    return data

def meanmc_g_iidstduniform(dimension, abs_tol, rel_tol):
    distribution = IIDStdUniform(dimension,seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = MeanMC_g(integrand, abs_tol, rel_tol).integrate()
    return data

def meanmc_g_iidstdgaussian(dimension, abs_tol, rel_tol):
    distribution = IIDStdGaussian(dimension,seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = MeanMC_g(integrand, abs_tol, rel_tol).integrate()
    return data

def cublattice_g(dimension, abs_tol, rel_tol):
    distribution = Lattice(dimension, seed=7, backend="GAIL")
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubLattice_g(integrand, abs_tol, rel_tol, check_cone=False).integrate()
    return data

def cubsobol_g(dimension, abs_tol, rel_tol):
    distribution = Sobol(dimension, seed=7, backend="MPS")
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubSobol_g(integrand, abs_tol, rel_tol, check_cone=False).integrate()
    return data

integrations_dict = {
    ('CLT','IIDStdUniform'): clt_iidstduniform,
    ('CLT','IIDStdGaussian'): clt_iidstdgaussian,
    ('CLTRep','Lattice'): cltrep_lattice,
    ('CLTRep','Sobol'): cltrep_sobol,
    ('MeanMC_g','IIDStdUniform'): meanmc_g_iidstduniform,
    ('MeanMC_g','IIDStdGaussian'): meanmc_g_iidstdgaussian,
    ('CubLattice_g','Lattice'): cublattice_g,
    ('CubSobol_g','Sobol'): cubsobol_g}