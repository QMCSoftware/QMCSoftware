import numpy as np
import numpy.typing as npt
import pandas as pd
import quantlib_util as qlu
import qmcpy_util as qpu
import config as cf


def add_theoretical_results(
    results_data: list, theoretical_mean: float, theoretical_std: float
) -> None:
    """
    Add theoretical benchmark values to results data.

    Args:
        results_data: List to append theoretical results to
        theoretical_mean: Theoretical expected value E[S_T]
        theoretical_std: Theoretical standard deviation of S_T
    """
    results_data.append(
        {
            "Method": "Theoretical",
            "Sampler": "-",
            "Mean": theoretical_mean,
            "Std Dev": theoretical_std,
            "Mean Absolute Error": 0,
            "Std Dev Error": 0,
        }
    )


def add_quantlib_results(
    results_data: list,
    sampler_type: str,
    quantlib_final: npt.NDArray[np.floating],  # per replication mean
    theoretical_mean: float,
    theoretical_std: float,
) -> None:
    ql_emp_mean = np.mean(quantlib_final)
    ql_emp_std = np.std(quantlib_final, ddof=1)
    ql_mae = np.mean(np.abs(quantlib_final - theoretical_mean))

    results_data.append(
        {
            "Method": "QuantLib",
            "Sampler": sampler_type,
            "Mean": ql_emp_mean,
            "Std Dev": ql_emp_std,
            "Mean Absolute Error": ql_mae,
            "Std Dev Error": abs(ql_emp_std - theoretical_std),
        }
    )


def add_qmcpy_results(
    results_data: list,
    sampler_type: str,
    qmcpy_final: npt.NDArray[np.floating],  # per replication mean
    qp_emp_mean: float,
    theoretical_mean: float,
    theoretical_std: float,
) -> None:
    qp_emp_std = np.std(qmcpy_final, ddof=1)
    qp_mae = np.mean(np.abs(qmcpy_final - theoretical_mean))

    results_data.append(
        {
            "Method": "QMCPy",
            "Sampler": sampler_type,
            "Mean": qp_emp_mean,
            "Std Dev": qp_emp_std,
            "Mean Absolute Error": qp_mae,
            "Std Dev Error": abs(qp_emp_std - theoretical_std),
        }
    )


# def process_sampler_data(
#     sampler_type: str,
#     results_data: list,
#     theoretical_mean: float,
#     theoretical_std: float,
#     params_ql: dict,
#     params_qp: dict,
# ) -> tuple:
#     """
#     Process and compare data for a single sampler type across both libraries.

#     Args:
#         sampler_type: Type of sampler to test
#         results_data: List to append comparison results to
#         theoretical_mean: Theoretical expected value
#         theoretical_std: Theoretical standard deviation
#         params_ql: Dictionary of QuantLib parameters
#         params_qp: Dictionary of QMCPy parameters

#     Returns:
#         tuple: (quantlib_paths, qmcpy_paths, ql_gbm, qp_gbm, params_ql, params_qp)
#     """
#     # Initialize quantlib_paths to None
#     params_ql["sampler_type"] = sampler_type
#     params_qp["sampler_type"] = sampler_type
#     quantlib_paths, quantlib_final = None, None

#     # Generate paths for both libraries
#     if sampler_type in ["IIDStdUniform", "Sobol"]:
#         quantlib_paths, ql_gbm = qlu.generate_quantlib_paths(**params_ql)
#         quantlib_final = quantlib_paths[:, -1]
#     else:
#         quantlib_paths, ql_gbm = None, None

#     qmcpy_paths, qp_gbm = qpu.generate_qmcpy_paths(**params_qp)

#     # Handle 3D array (replications, n_paths, n_steps) by taking first replication
#     if qmcpy_paths.ndim == 3:
#         qmcpy_paths = qmcpy_paths[0]

#     # Final value statistics
#     qmcpy_final = qmcpy_paths[:, -1]
#     qp_emp_mean = np.mean(qmcpy_final)

#     # Add results to data
#     if sampler_type in ["IIDStdUniform", "Sobol"] and quantlib_final is not None:
#         add_quantlib_results(
#             results_data,
#             sampler_type,
#             quantlib_final,
#             theoretical_mean,
#             theoretical_std,
#         )
#     add_qmcpy_results(
#         results_data,
#         sampler_type,
#         qmcpy_final,
#         qp_emp_mean,
#         theoretical_mean,
#         theoretical_std,
#     )

#     return quantlib_paths, qmcpy_paths, ql_gbm, qp_gbm, params_ql, params_qp


def process_sampler_data(
    sampler_type: str,
    results_data: list,
    theoretical_mean: float,
    theoretical_std: float,
    params_ql: dict,
    params_qp: dict,
) -> tuple:
    """
    Process and compare data for a single sampler type across both libraries.

    Args:
        sampler_type: Type of sampler to test
        results_data: List to append comparison results to
        theoretical_mean: Theoretical expected value
        theoretical_std: Theoretical standard deviation
        params_ql: Dictionary of QuantLib parameters
        params_qp: Dictionary of QMCPy parameters

    Returns:
        tuple: (quantlib_paths, qmcpy_paths, ql_gbm, qp_gbm, params_ql, params_qp)
    """

    params_ql["sampler_type"] = sampler_type
    params_qp["sampler_type"] = sampler_type

    replications = params_qp["replications"]

    quantlib_paths, ql_gbm = None, None

    if sampler_type in ["IIDStdUniform", "Sobol"]:
        ql_means = np.empty(replications)
        ql_seed = params_ql["seed"]

        for r in range(replications):
            params_ql["seed"] = ql_seed + r
            quantlib_paths, ql_gbm = qlu.generate_quantlib_paths(**params_ql)
            ql_means[r] = quantlib_paths[:, -1].mean()

        params_ql["seed"] = ql_seed
    else:
        ql_means = None

    qmcpy_paths, qp_gbm = qpu.generate_qmcpy_paths(**params_qp)

    if qmcpy_paths.ndim == 3:
        qp_means = qmcpy_paths[:, :, -1].mean(axis=1)
    else:
        qp_means = np.array([qmcpy_paths[:, -1].mean()])

    if ql_means is not None:
        add_quantlib_results(
            results_data,
            sampler_type,
            ql_means,
            theoretical_mean,
            theoretical_std,
        )

    add_qmcpy_results(
        results_data,
        sampler_type,
        qp_means,
        qp_means.mean(),
        theoretical_mean,
        theoretical_std,
    )

    return quantlib_paths, qmcpy_paths, ql_gbm, qp_gbm, params_ql, params_qp


def create_timing_dataframe(
    quantlib_results: dict, qmcpy_results: dict, baseline_sampler: str
) -> pd.DataFrame:
    """
    Create comprehensive timing comparison table from benchmark results.

    Args:
        quantlib_results: Dictionary mapping sampler names to timing results
        qmcpy_results: Dictionary mapping sampler names to timing results
        baseline_sampler: Sampler to use as baseline for speedup calculation

    Returns:
        DataFrame with timing statistics and speedup comparisons
    """
    timing_data = []

    # Add QuantLib data
    for sampler_type, result in quantlib_results.items():
        timing_data.append(
            {
                "Method": "QuantLib",
                "Sampler": sampler_type,
                "Mean Time (s)": result["average"],
                "Std Dev (s)": result["stdev"],
                "Speedup": "-",
            }
        )

    # Add QMCPy data with speedup calculation
    baseline_time = quantlib_results[baseline_sampler]["average"]
    for sampler_type, result in qmcpy_results.items():
        speedup = baseline_time / result["average"]
        timing_data.append(
            {
                "Method": "QMCPy",
                "Sampler": sampler_type,
                "Mean Time (s)": result["average"],
                "Std Dev (s)": result["stdev"],
                "Speedup": speedup,
            }
        )

    return pd.DataFrame(timing_data)


def extract_comparison_data(results_df: pd.DataFrame) -> tuple:
    """
    Extract data for comparison plotting from results dataframe.

    Args:
        results_df: DataFrame containing results from both libraries

    Returns:
        tuple: (samplers, qmcpy_errors, qmcpy_times, quantlib_errors,
                quantlib_times, theoretical_mean)
    """
    qmcpy_data = results_df[results_df["Method"] == "QMCPy"].copy()
    quantlib_data = results_df[results_df["Method"] == "QuantLib"].copy()
    theoretical_data = results_df[results_df["Method"] == "Theoretical"].copy()

    samplers = qmcpy_data["Sampler"].values
    qmcpy_errors = qmcpy_data["Mean Absolute Error"].values
    qmcpy_times = (
        qmcpy_data["Mean Time (s)"].values
        if "Mean Time (s)" in qmcpy_data.columns
        else None
    )

    # Get QuantLib data (only available for some samplers
    ql_error_dict = dict(
        zip(quantlib_data["Sampler"], quantlib_data["Mean Absolute Error"])
    )
    quantlib_errors = [ql_error_dict.get(s) for s in samplers]

    if "Mean Time (s)" in quantlib_data.columns:
        ql_time_dict = dict(
            zip(quantlib_data["Sampler"], quantlib_data["Mean Time (s)"])
        )
        quantlib_times = [ql_time_dict.get(s) for s in samplers]
    else:
        quantlib_times = [None] * len(samplers)

    # Handle case where theoretical data might be missing
    if not theoretical_data.empty:
        theoretical_mean = theoretical_data["Mean"].iloc[0]
    else:
        # Calculate theoretical mean from parameters if not in results_df
        # Using the parameters from the comparison study
        S0, mu, T = 100, 0.05, 1.0
        theoretical_mean = S0 * np.exp(mu * T)

    return (
        samplers,
        qmcpy_errors,
        qmcpy_times,
        quantlib_errors,
        quantlib_times,
        theoretical_mean,
    )


def add_theoretical_row(
    results: list,
    series_name: str,
    n_steps: int,
    n_paths: int,
    theoretical_mean: float,
    theoretical_std: float,
) -> None:
    """Add theoretical benchmark row to results"""
    results.append(
        {
            "Series": series_name,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "Method": "Theoretical",
            "Sampler": "-",
            "Mean": theoretical_mean,
            "Std Dev": theoretical_std,
            "Mean Absolute Error": 0,
            "Std Dev Error": 0,
            "Runtime (s)": 0,
            "Runtime Std (s)": 0,
        }
    )


def collect_library_results(
    sampler: str,
    series_name: str,
    n_steps: int,
    n_paths: int,
    ql_timing: dict,
    qp_timing: dict,
    theoretical_mean: float,
    theoretical_std: float,
) -> list:
    """Collect results for both QuantLib and QMCPy for a given sampler"""
    results = []
    gbm_params = cf.get_gbm_parameters()

    # QuantLib parameters
    ql_params = {**gbm_params, "n_steps": n_steps, "n_paths": n_paths}

    # QMCPy parameters (note: diffusion = sigma^2)
    qp_params = {
        "initial_value": gbm_params["initial_value"],
        "mu": gbm_params["mu"],
        "diffusion": gbm_params["sigma"] ** 2,  # Convert sigma to diffusion
        "maturity": gbm_params["maturity"],
        "n_steps": n_steps,
        "n_paths": n_paths,
    }

    # QuantLib results (if supported)
    if sampler in cf.get_sampler_configurations()["quantlib_samplers"]:
        try:
            ql_paths, _ = qlu.generate_quantlib_paths(sampler_type=sampler, **ql_params)
            ql_final = ql_paths[:, -1]
            ql_mean = np.mean(ql_final)
            ql_std = np.std(ql_final, ddof=1)

            results.append(
                {
                    "Series": series_name,
                    "n_steps": n_steps,
                    "n_paths": n_paths,
                    "Method": "QuantLib",
                    "Sampler": sampler,
                    "Mean": ql_mean,
                    "Std Dev": ql_std,
                    "Mean Absolute Error": abs(ql_mean - theoretical_mean),
                    "Std Dev Error": abs(ql_std - theoretical_std),
                    "Runtime (s)": ql_timing[sampler]["average"],
                    "Runtime Std (s)": ql_timing[sampler]["stdev"],
                }
            )
        except Exception as e:
            print(f"      QuantLib {sampler} failed: {e}")

    # QMCPy results
    try:
        qp_paths, _ = qpu.generate_qmcpy_paths(sampler_type=sampler, **qp_params)
        qp_final = qp_paths[:, -1]
        qp_mean = np.mean(qp_final)
        qp_std = np.std(qp_final, ddof=1)

        results.append(
            {
                "Series": series_name,
                "n_steps": n_steps,
                "n_paths": n_paths,
                "Method": "QMCPy",
                "Sampler": sampler,
                "Mean": qp_mean,
                "Std Dev": qp_std,
                "Mean Absolute Error": abs(qp_mean - theoretical_mean),
                "Std Dev Error": abs(qp_std - theoretical_std),
                "Runtime (s)": qp_timing[sampler]["average"],
                "Runtime Std (s)": qp_timing[sampler]["stdev"],
            }
        )
    except Exception as e:
        print(f"      QMCPy {sampler} failed: {e}")

    return results
