import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from qmcpy_util_replications import generate_qmcpy_paths
import config as cf
import plot_util as pu
from matplotlib.ticker import FixedLocator, FixedFormatter

def compute_mae_vs_paths(sampler_type='Sobol',
                                replications=5):
    """
    Compute mean absolute error vs number of paths using independent QMCPy replications
    """

    # Get experiment configuration 
    exp_cfg = cf.get_experiment_configurations()['paths']
    n_paths_range = exp_cfg['range']
    n_steps = exp_cfg['fixed_steps']

    # Get GBM parameters
    gbm_params = cf.get_gbm_parameters()
    initial_value = gbm_params['initial_value']
    mu = gbm_params['mu']
    diffusion = gbm_params['sigma']**2 # diffusion = sigma^2
    maturity = gbm_params['maturity']

    # Compute theoretical mean
    theoretical_mean = initial_value * np.exp(mu * maturity)
    
    mean_errors = []

    for n_paths in n_paths_range:
        # Generate GBM paths (with idependant replications) for each number of paths in n_paths_range
        paths, gbm = generate_qmcpy_paths(
            initial_value=initial_value,
            mu=mu,
            diffusion=diffusion,
            maturity=maturity,
            n_steps=n_steps,
            n_paths=n_paths,
            sampler_type=sampler_type,
            replications=replications,
        )

        # Extract values at the final time
        final_vals = paths[:, :, -1]  

        # Compute MAE for each replication and take the average
        errors_per_repl = np.abs(final_vals.mean(axis=1) - theoretical_mean)
        mean_errors.append(errors_per_repl.mean())

    return n_paths_range, mean_errors

def plot_mae_vs_paths(replications):

    styling = pu.get_plot_styling()
    colors = styling['colors']['QMCPy']
    markers = styling['markers']['QMCPy']
    samplers = cf.get_sampler_configurations()['all_samplers'] 

    fig, ax = plt.subplots(figsize=(10,6))

    for sampler in samplers:
        n_paths_range, mean_errors = compute_mae_vs_paths(sampler_type=sampler, replications=replications)
        
        ax.loglog(
            n_paths_range,
            mean_errors,
            marker=markers.get(sampler),
            color=colors.get(sampler),
            linewidth=2,
            markersize=6,
            label=sampler
        )

    ax.xaxis.set_major_locator(FixedLocator(n_paths_range))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(x)) for x in n_paths_range]))
    ax.xaxis.set_minor_locator(FixedLocator([]))  


    ax.set_xlabel("Number of Paths", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Absolute Error", fontsize=12, fontweight='bold')
    ax.set_title(f"MAE vs Number of Paths across {replications} replications (n_steps = 252)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.show()

    #old version
def compute_mean_error_vs_paths(n_paths_list, sampler_type='Sobol', n_steps=252,
                                replications=5, initial_value=100, mu=0.05,
                                diffusion=0.2**2, maturity=1.0, seed=42):
    """
    Compute mean absolute error vs number of paths using independent QMCPy replications.
    """


    theo_mean = initial_value * np.exp(mu * maturity)
    mean_errors = []

    for n_paths in n_paths_list:
        # Generate paths with replications
        paths, gbm = generate_qmcpy_paths(
            initial_value=initial_value,
            mu=mu,
            diffusion=diffusion,
            maturity=maturity,
            n_steps=n_steps,
            n_paths=n_paths,
            sampler_type=sampler_type,
            replications=replications,
            seed=seed
        )

        final_vals = paths[:, :, -1]

        errors_per_repl = np.abs(final_vals.mean(axis=1) - theo_mean)

        mean_errors.append(errors_per_repl.mean())

    return mean_errors
