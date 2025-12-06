is_debug = False

def get_experiment_configurations():
    """Define experimental parameter ranges"""
    return {
        'time_steps': {
            'fixed_paths': 2**3 if is_debug else 2**12,
            'range': [2**i for i in range(4, 7)] if is_debug else [2**i for i in range(4, 10)],  # [16, 32, 64, 128, 256, 512]
            'series_name': 'Time Steps'
        },
        'paths': {
            'fixed_steps':  2**5 if is_debug else 252,
            'range': [2**i for i in range(6, 9)] if is_debug else [2**i for i in range(9, 15)],  # [256, ..., 131072]
            'series_name': 'Paths'
        }
    }

def get_sampler_configurations():
    """Define sampler types for testing"""
    return {
        'all_samplers': ['IIDStdUniform', 'Sobol', 'Lattice', 'Halton'],
        'quantlib_samplers': ['IIDStdUniform', 'Sobol']
    }

def get_gbm_parameters():
    """Define base GBM parameters"""
    return {
        'initial_value': 100,
        'mu': 0.05,
        'sigma': 0.2,
        'maturity': 1.0
    }

