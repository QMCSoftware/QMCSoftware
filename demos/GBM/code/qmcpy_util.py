import qmcpy as qp

def create_qmcpy_sampler(sampler_type, dimension, seed=42, replications):
    """Create a sampler instance based on type and dimension"""
    if sampler_type == 'IIDStdUniform':
        return qp.IIDStdUniform(dimension, seed=seed,replications)
    elif sampler_type == 'Sobol':
        return qp.Sobol(dimension, seed=seed,replications)
    elif sampler_type == 'Lattice':
        return qp.Lattice(dimension, seed=seed,replications)
    elif sampler_type == 'Halton':
        return qp.Halton(dimension, seed=seed,replications)
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")
    

def generate_qmcpy_paths(initial_value, mu, diffusion, maturity, n_steps, n_paths, sampler_type='IIDStdUniform', seed=42):
    """Generate GBM paths using QMCPy with configurable sampler."""
    sampler = create_qmcpy_sampler(sampler_type, n_steps, seed)
    gbm = qp.GeometricBrownianMotion(sampler, t_final=maturity, initial_value=initial_value, drift=mu, diffusion=diffusion)
    paths = gbm.gen_samples(n_paths)
    return paths, gbm

