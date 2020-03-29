""" Integrations problems to vary parameters on """

from qmcpy import *

def clt_iidstduniform(dimension, abs_tol, time_vector, mean_shift_is):
    distribution = IIDStdUniform(dimension,seed=7)
    measure = BrownianMotion(distribution, time_vector, mean_shift_is)
    integrand = AsianCall(measure)
    solution,data = CLT(integrand, abs_tol).integrate()
    return data

def clt_iidstdgaussian(dimension, abs_tol, time_vector, mean_shift_is):
    distribution = IIDStdGaussian(dimension,seed=7)
    measure = BrownianMotion(distribution, time_vector, mean_shift_is)
    integrand = AsianCall(measure)
    solution,data = CLT(integrand, abs_tol).integrate()
    return data

def meanmc_g_iidstduniform(dimension, abs_tol, time_vector, mean_shift_is):
    distribution = IIDStdUniform(dimension,seed=7)
    measure = BrownianMotion(distribution, time_vector, mean_shift_is)
    integrand = AsianCall(measure)
    solution,data = MeanMC_g(integrand, abs_tol).integrate()
    return data

def meanmc_g_iidstdgaussian(dimension, abs_tol, time_vector, mean_shift_is):
    distribution = IIDStdGaussian(dimension,seed=7)
    measure = BrownianMotion(distribution, time_vector, mean_shift_is)
    integrand = AsianCall(measure)
    solution,data = MeanMC_g(integrand, abs_tol).integrate()
    return data

def cltrep_lattice(dimension, abs_tol, time_vector, mean_shift_is):
    distribution = Lattice(dimension, replications=16, seed=7, backend="MPS")
    measure = BrownianMotion(distribution, time_vector, mean_shift_is)
    integrand = AsianCall(measure)
    solution,data = CLTRep(integrand, abs_tol).integrate()
    return data

def cltrep_sobol(dimension, abs_tol, time_vector, mean_shift_is):
    distribution = Sobol(dimension, replications=16, seed=7, backend="MPS")
    measure = BrownianMotion(distribution, time_vector, mean_shift_is)
    integrand = AsianCall(measure)
    solution,data = CLTRep(integrand, abs_tol).integrate()
    return data

def cublattice_g(dimension, abs_tol, time_vector, mean_shift_is):
    distribution = Lattice(dimension, seed=7, backend="GAIL")
    measure = BrownianMotion(distribution, time_vector, mean_shift_is)
    integrand = AsianCall(measure)
    solution,data = CubLattice_g(integrand, abs_tol, check_cone=False).integrate()
    return data

def cubsobol_g(dimension, abs_tol, time_vector, mean_shift_is):
    distribution = Sobol(dimension, seed=7, backend="MPS")
    measure = BrownianMotion(distribution, time_vector, mean_shift_is)
    integrand = AsianCall(measure)
    solution,data = CubSobol_g(integrand, abs_tol, check_cone=False).integrate()
    return data

integrations_dict = {
    ('CLT','IIDStdUniform','MC'): clt_iidstduniform,
    ('CLT','IIDStdGaussian','MC'): clt_iidstdgaussian,
    ('MeanMC_g','IIDStdUniform','MC'): meanmc_g_iidstduniform,
    ('MeanMC_g','IIDStdGaussian','MC'): meanmc_g_iidstdgaussian,
    ('CLTRep','Lattice','QMC'): cltrep_lattice,
    ('CLTRep','Sobol','QMC'): cltrep_sobol,
    ('CubLattice_g','Lattice','QMC'): cublattice_g,
    ('CubSobol_g','Sobol','QMC'): cubsobol_g}