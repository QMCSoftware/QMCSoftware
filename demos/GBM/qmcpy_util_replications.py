import qmcpy as qp

def create_qmcpy_sampler(sampler_type, dimension, replications, seed=42):
    """Create a sampler instance based on type and dimension"""
    if sampler_type == 'IIDStdUniform':
        return qp.IIDStdUniform(dimension, replications, seed=seed)
    elif sampler_type == 'Sobol':
        return qp.Sobol(dimension, replications, seed=seed)
    elif sampler_type == 'Lattice':
        return qp.Lattice(dimension, replications, seed=seed)
    elif sampler_type == 'Halton':
        return qp.Halton(dimension, replications, seed=seed)
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


def generate_qmcpy_paths(initial_value, mu, diffusion, maturity, n_steps,
                         n_paths, sampler_type='IIDStdUniform',
                         replications=1, seed=42):
    """Generate GBM paths using QMCPy with configurable sampler."""
    sampler = create_qmcpy_sampler(
        sampler_type, n_steps, replications, seed)
    gbm = qp.GeometricBrownianMotion(
        sampler, t_final=maturity, initial_value=initial_value,
        drift=mu, diffusion=diffusion)
    paths = gbm.gen_samples(n_paths)
    return paths, gbm