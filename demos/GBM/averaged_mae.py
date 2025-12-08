import numpy as np
import matplotlib.pyplot as plt
import quantlib_util as qlu
import qmcpy_util_replications as qpru
import config as cf
import plot_util as pu
from matplotlib.ticker import FixedLocator, FixedFormatter

def compute_mae_vs_paths(method, sampler, replications=5, qp_seed=42, ql_seed=7):
    """
    Compute averaged MAE vs number of paths for all samplers.
    
    Args:
        method: Either "QMCPy" or "QuantLib"
        sampler: Sampler type (e.g., 'IIDStdUniform', 'Sobol', 'Lattice', 'Halton')
        replications: Number of independent replications to average over
        qp_seed: Random seed for QMCPy samplers
        ql_seed: Random seed for QuantLib samplers
        
    Returns:
        tuple: (n_paths_range, mean_errors) - path counts and corresponding errors
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
            seed = qp_seed
            paths, _ = qpru.generate_qmcpy_paths(
                initial_value=initial_value,
                mu=mu,
                diffusion=diffusion,
                maturity=maturity,
                n_steps=n_steps,
                n_paths=n_paths,
                sampler_type=sampler,
                replications=replications,
                seed=seed
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
    all_pairs = [("QuantLib", s) for s in ql_samplers] + [("QMCPy", s) for s in qmcpy_samplers]

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



def compute_mae_vs_steps(method, sampler, replications=5, qp_seed=42, ql_seed=7):
    """
    Compute averaged MAE vs number of steps for all samplers
    """
    # Experiment configuration
    exp_cfg = cf.get_experiment_configurations()['time_steps']
    n_steps_range = exp_cfg['range']
    n_paths = exp_cfg['fixed_paths']  # fixed number of paths

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

    for n_steps in n_steps_range:
        if method == "QMCPy":
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
            final_vals = paths[:, :, -1]  
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

    return n_steps_range, mean_errors


def plot_mae_vs_steps(replications=5):
    """
    Plot averaged MAE vs number of steps for all samplers
    """
    styling = pu.get_plot_styling()
    sampler_cfg = cf.get_sampler_configurations()
    qmcpy_samplers = sampler_cfg['all_samplers']
    ql_samplers = sampler_cfg['quantlib_samplers']

    all_pairs = [("QuantLib", s) for s in ql_samplers] + [("QMCPy", s) for s in qmcpy_samplers]

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, sampler in all_pairs:
        n_steps_range, mean_errors = compute_mae_vs_steps(method, sampler, replications)

        colors = styling['colors'][method]
        markers = styling['markers'][method]

        ax.loglog(
            n_steps_range,
            mean_errors,
            marker=markers.get(sampler),
            color=colors.get(sampler),
            linewidth=2,
            markersize=6,
            label=f"{method} - {sampler}"
        )

    ax.xaxis.set_major_locator(FixedLocator(n_steps_range))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(x)) for x in n_steps_range]))
    ax.xaxis.set_minor_locator(FixedLocator([]))

    ax.set_xlabel("Number of Steps", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Absolute Error", fontsize=12, fontweight='bold')
    ax.set_title(f"MAE vs Number of Time Steps across {replications} replications (n_paths = 4,096)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.savefig('images/figure_8.png', bbox_inches='tight', dpi=150)
    plt.show()

def update_sweep_df(df, replications=5):
    """
    Replace values in 'Mean Absolute Error' from sweep_results_df with averaged MAE values computed
    using compute_mae_vs_paths and compute_mae_vs_steps.
    """
    df = df.copy()
    
    # Get sampler configurations
    exp_cfg = cf.get_experiment_configurations()
    ql_samplers = cf.get_sampler_configurations()['quantlib_samplers']
    qp_samplers = cf.get_sampler_configurations()['all_samplers']

    # Compute MAE for all (method, sampler) pairs
    mae_vals = {}

    # Compute MAE vs number of paths
    for method in ["QMCPy", "QuantLib"]:
        samplers_to_loop = qp_samplers if method == "QMCPy" else ql_samplers
        for sampler in samplers_to_loop:
            x, y = compute_mae_vs_paths(method, sampler, replications=replications)
            mae_vals[(method, sampler, "paths")] = dict(zip(x, y))

    # Compute MAE vs number of time steps
    for method in ["QMCPy", "QuantLib"]:
        samplers_to_loop = qp_samplers if method == "QMCPy" else ql_samplers
        for sampler in samplers_to_loop:
            x, y = compute_mae_vs_steps(method, sampler, replications=replications)
            mae_vals[(method, sampler, "steps")] = dict(zip(x, y))

    # Update the dataframe

    # Keep theoretical values
    for idx, row in df.iterrows():
        if row["Method"] == "Theoretical":
            continue

        method = row["Method"]
        sampler = row["Sampler"]

    # Update MAEs for chaning number of paths
        if row["Series"] == "Paths":
            lookup_key = (method, sampler, "paths")
            if lookup_key in mae_vals:
                df.loc[idx, "Mean Absolute Error"] = mae_vals[lookup_key][row["n_paths"]]

     # Update MAEs for chaning number of time steps
        elif row["Series"] == "Time Steps":
            lookup_key = (method, sampler, "steps")
            if lookup_key in mae_vals:
                df.loc[idx, "Mean Absolute Error"] = mae_vals[lookup_key][row["n_steps"]]

    return df
