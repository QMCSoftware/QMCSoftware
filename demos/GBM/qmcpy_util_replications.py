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
    """
    Generate Geometric Brownian Motion paths using QMCPy with multiple replications.
    
    Args:
        initial_value: Initial value of the GBM process (S_0)
        mu: Drift parameter
        diffusion: Diffusion coefficient (sigma^2)
        maturity: Final time T
        n_steps: Number of discretization time steps
        n_paths: Number of paths to generate per replication
        sampler_type: Type of sampler ('IIDStdUniform', 'Sobol', 'Lattice', 'Halton')
        replications: Number of independent replications
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (paths, gbm) where paths has shape (replications, n_paths, n_steps)
               and gbm is the GeometricBrownianMotion object
    """
    sampler = create_qmcpy_sampler(
        sampler_type, n_steps, replications, seed)
    gbm = qp.GeometricBrownianMotion(
        sampler, t_final=maturity, initial_value=initial_value,
        drift=mu, diffusion=diffusion)
    paths = gbm.gen_samples(n_paths)
    return paths, gbm