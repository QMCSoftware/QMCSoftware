import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from quantlib_util import generate_quantlib_paths
import config as cf
import plot_util as pu
from matplotlib.ticker import FixedLocator, FixedFormatter


def compute_ql_mae_vs_paths(
        sampler_type='Sobol',
        replications=5,
        ql_seed=7): 
    """
    Compute averaged MAE vs number of paths for QuantLib samplers.
    """

   # Get experiment configuration 
    exp_cfg = cf.get_experiment_configurations()['paths']
    n_paths_range = exp_cfg['range']
    n_steps = exp_cfg['fixed_steps']

    # Get GBM parameters
    gbm_params = cf.get_gbm_parameters()
    initial_value = gbm_params['initial_value']
    mu = gbm_params['mu']
    sigma = gbm_params['sigma']
    maturity = gbm_params['maturity']

    # Compute theoretical mean
    theoretical_mean = initial_value * np.exp(mu * maturity)

    mean_errors = []

    for n_paths in n_paths_range:
        errors = []

        
        for r in range(replications):
            seed = ql_seed + r

            paths, gbm = generate_quantlib_paths(
                initial_value=initial_value,
                mu=mu,
                sigma=sigma,
                maturity=maturity,
                n_steps=n_steps,
                n_paths=n_paths,
                sampler_type=sampler_type,
                seed=seed
            )

            
            final_vals = paths[:, -1]

            # MAE for this replication
            errors.append(np.abs(final_vals.mean() - theoretical_mean))

        # Average across replications
        mean_errors.append(np.mean(errors))

    return n_paths_range, mean_errors


def plot_ql_mae_vs_paths(replications):
    styling = pu.get_plot_styling()
    colors = styling['colors']['QuantLib']
    markers = styling['markers']['QuantLib']
    samplers = cf.get_sampler_configurations()['quantlib_samplers']

    fig, ax = plt.subplots(figsize=(10,6))

    for sampler in samplers:
        n_paths_range, mean_errors = compute_ql_mae_vs_paths(sampler_type=sampler, replications=replications)
        
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
