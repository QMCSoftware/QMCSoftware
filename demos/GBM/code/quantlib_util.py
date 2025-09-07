import QuantLib as ql
import numpy as np

def generate_quantlib_paths(initial_value, mu, sigma, maturity, n_steps, n_paths, sampler_type='IIDStdUniform', seed=7):
    """Generate GBM paths using QuantLib with configurable sampler type and seed."""
    gbm = ql.GeometricBrownianMotionProcess(initial_value, mu, sigma)
    times = ql.TimeGrid(maturity, n_steps)
    dimension = n_steps
    if sampler_type == 'IIDStdUniform':
        uniform_rng = ql.UniformRandomGenerator(seed)
        sequence_gen = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(n_steps, uniform_rng))
        path_gen = ql.GaussianPathGenerator(gbm, maturity, n_steps, sequence_gen, False)
        paths = np.zeros((n_paths, n_steps + 1))
        for i in range(n_paths):
            sample_path = path_gen.next().value()
            paths[i, :] = np.array([sample_path[j] for j in range(n_steps + 1)])
        return paths, gbm
    elif sampler_type == 'Sobol':
        uniform_rsg = ql.UniformLowDiscrepancySequenceGenerator(dimension, seed)
        sequence_gen = ql.GaussianLowDiscrepancySequenceGenerator(uniform_rsg)
        path_gen = ql.GaussianSobolMultiPathGenerator(gbm, list(times), sequence_gen, False)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = initial_value  # Set initial value
        for i in range(n_paths):
            sample_path = path_gen.next().value()
            path_values = sample_path[0]  # For 1D process, get the first (and only) path
            paths[i, :] = np.array([path_values[j] for j in range(n_steps + 1)])
        return paths, gbm
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}. Use 'IIDStdUniform' or 'Sobol'")
    

