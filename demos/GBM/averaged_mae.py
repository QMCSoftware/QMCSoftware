import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import quantlib_util as qlu
import qmcpy_util_replications as qpru
import qmcpy_util as qpu
import config as cf
import plot_util as pu
from matplotlib.ticker import FixedLocator, FixedFormatter

def compute_mae_vs_paths(method, sampler, replications=5, qp_seed = 42, ql_seed=7):
    """
    Compute averaged MAE vs number of paths for all samplers
    """
    # Experiment configuration
    exp_cfg = cf.get_experiment_configurations()['paths']
    n_paths_range = exp_cfg['range']
    n_steps = exp_cfg['fixed_steps']

    # GBM parameters
    gbm_params = cf.get_gbm_parameters()
    initial_value = gbm_params['initial_value']
    mu = gbm_params['mu']
    sigma = gbm_params['sigma']
    diffusion = sigma**2
    maturity = gbm_params['maturity']

    # Theoretical mean
    theoretical_mean = initial_value * np.exp(mu * maturity)

    mean_errors = []

    for n_paths in n_paths_range:
        if method == "QMCPy":
            # errors = []
            # for r in range(replications):
            #     seed = qp_seed + r
            #     paths, _ = qpu.generate_qmcpy_paths(
            #         initial_value = initial_value,
            #         mu=mu,
            #         diffusion=diffusion,
            #         maturity=maturity,
            #         n_steps=n_steps,
            #         n_paths=n_paths,
            #         sampler_type=sampler,
            #         seed=seed
            #     )
            #     final_vals = paths[:, -1]
            #     errors.append(np.abs(final_vals.mean() - theoretical_mean))

            paths, _ = qpru.generate_qmcpy_paths(
                initial_value=initial_value,
                mu=mu,
                diffusion=diffusion,
                maturity=maturity,
                n_steps=n_steps,
                n_paths=n_paths,
                sampler_type=sampler,
                replications=replications
            )
            # Extract values at the final time
            final_vals = paths[:, :, -1]  
            # Compute MAE for each replication and take the average
            errors = np.abs(final_vals.mean(axis=1) - theoretical_mean)

        elif method == "QuantLib":
            errors = []
            for r in range(replications):
                seed = ql_seed + r
                paths, _ = qlu.generate_quantlib_paths(
                    initial_value=initial_value,
                    mu=mu,
                    sigma=sigma,
                    maturity=maturity,
                    n_steps=n_steps,
                    n_paths=n_paths,
                    sampler_type=sampler,
                    seed=seed
                )
                final_vals = paths[:, -1]
                errors.append(np.abs(final_vals.mean() - theoretical_mean))

        mean_errors.append(np.mean(errors))

    return n_paths_range, mean_errors


def plot_mae_vs_paths(replications=5):
    """
    Plot averaged MAE vs number of paths for all samplers
    """
    styling = pu.get_plot_styling()
    sampler_cfg = cf.get_sampler_configurations()
    qmcpy_samplers = sampler_cfg['all_samplers']
    ql_samplers = sampler_cfg['quantlib_samplers']

    # list of (method, sample) pairs to plot
    all_pairs = [("QMCPy", s) for s in qmcpy_samplers] + [("QuantLib", s) for s in ql_samplers]

    fig, ax = plt.subplots(figsize=(10, 6))

    for method, sampler in all_pairs:
        n_paths_range, mean_errors = compute_mae_vs_paths(method, sampler, replications)

        colors = styling['colors'][method]
        markers = styling['markers'][method]

        ax.loglog(
            n_paths_range,
            mean_errors,
            marker=markers.get(sampler),
            color=colors.get(sampler),
            linewidth=2,
            markersize=6,
            label=f"{method} - {sampler}"
        )

    ax.xaxis.set_major_locator(FixedLocator(n_paths_range))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(x)) for x in n_paths_range]))
    ax.xaxis.set_minor_locator(FixedLocator([]))

    ax.set_xlabel("Number of Paths", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Absolute Error", fontsize=12, fontweight='bold')
    ax.set_title(f"MAE vs Number of Paths across {replications} replications (n_steps = 252)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.show()
