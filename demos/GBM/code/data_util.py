import numpy as np
import pandas as pd
import quantlib_util as qlu
import qmcpy_util as qpu
import config as cf

def add_theoretical_results(results_data, theoretical_mean, theoretical_std):
    """Add theoretical values to results data"""
    results_data.append({
        'Method': 'Theoretical',
        'Sampler': '-',
        'Mean': theoretical_mean,
        'Std Dev': theoretical_std,
        'Mean Absolute Error': 0,
        'Std Dev Error': 0,
    })

def add_quantlib_results(results_data, sampler_type, quantlib_final, theoretical_mean, theoretical_std):
    """Add QuantLib results to results data"""
    ql_emp_mean = np.mean(quantlib_final)
    ql_emp_std = np.std(quantlib_final, ddof=1)
    results_data.append({
        'Method': 'QuantLib',
        'Sampler': sampler_type,
        'Mean': ql_emp_mean,
        'Std Dev': ql_emp_std,
        'Mean Absolute Error': abs(ql_emp_mean - theoretical_mean),
        'Std Dev Error': abs(ql_emp_std - theoretical_std),
    })

def add_qmcpy_results(results_data, sampler_type, qmcpy_final, qp_emp_mean, theoretical_mean, theoretical_std):
    """Add QMCPy results to results data"""
    qp_emp_std = np.std(qmcpy_final, ddof=1)
    results_data.append({
        'Method': 'QMCPy',
        'Sampler': sampler_type,
        'Mean': qp_emp_mean,
        'Std Dev': qp_emp_std,
        'Mean Absolute Error': abs(qp_emp_mean - theoretical_mean),
        'Std Dev Error': abs(qp_emp_std - theoretical_std),
    })

def process_sampler_data(sampler_type, results_data, theoretical_mean, theoretical_std, params_ql, params_qp):
    """Process data for a single sampler type"""
    # Initialize quantlib_paths to None
    params_ql['sampler_type'] = sampler_type
    params_qp['sampler_type'] = sampler_type
    quantlib_paths, quantlib_final = None, None

    # Generate paths for both libraries
    if sampler_type in ['IIDStdUniform', 'Sobol']:
        quantlib_paths, ql_gbm = qlu.generate_quantlib_paths(**params_ql)
        quantlib_final = quantlib_paths[:, -1]
    else: quantlib_paths, ql_gbm = None, None
        
    qmcpy_paths, qp_gbm = qpu.generate_qmcpy_paths(**params_qp)

    # Final value statistics
    qmcpy_final = qmcpy_paths[:, -1]
    qp_emp_mean = np.mean(qmcpy_final)

    # Add results to data
    if sampler_type in ['IIDStdUniform', 'Sobol'] and quantlib_final is not None:
        add_quantlib_results(results_data, sampler_type, quantlib_final, theoretical_mean, theoretical_std)
    add_qmcpy_results(results_data, sampler_type, qmcpy_final, qp_emp_mean, theoretical_mean, theoretical_std)
    
    return quantlib_paths, qmcpy_paths, ql_gbm, qp_gbm, params_ql, params_qp


def create_timing_dataframe(quantlib_results, qmcpy_results, baseline_sampler):
    """Create comprehensive timing table from benchmark results"""
    timing_data = []
    
    # Add QuantLib data
    for sampler_type, result in quantlib_results.items():
        timing_data.append({
            'Method': 'QuantLib',
            'Sampler': sampler_type,
            'Mean Time (s)': result['average'],
            'Std Dev (s)': result['stdev'],
            'Speedup': '-'
        })
    
    # Add QMCPy data with speedup calculation
    baseline_time = quantlib_results[baseline_sampler]['average']
    for sampler_type, result in qmcpy_results.items():
        speedup = baseline_time / result['average']
        timing_data.append({
            'Method': 'QMCPy',
            'Sampler': sampler_type,
            'Mean Time (s)': result['average'],
            'Std Dev (s)': result['stdev'],
            'Speedup': speedup
        })
    
    return pd.DataFrame(timing_data)

def extract_comparison_data(results_df):
    """Extract data for comparison plotting from results dataframe"""
    qmcpy_data = results_df[results_df['Method'] == 'QMCPy'].copy()
    quantlib_data = results_df[results_df['Method'] == 'QuantLib'].copy()
    theoretical_data = results_df[results_df['Method'] == 'Theoretical'].copy()
    
    samplers = qmcpy_data['Sampler'].values
    qmcpy_errors = qmcpy_data['Mean Absolute Error'].values
    qmcpy_times = qmcpy_data['Mean Time (s)'].values if 'Mean Time (s)' in qmcpy_data.columns else None
    
    # Get QuantLib data (only available for some samplers)
    quantlib_errors = []
    quantlib_times = []
    for sampler in samplers:
        ql_row = quantlib_data[quantlib_data['Sampler'] == sampler]
        if not ql_row.empty:
            quantlib_errors.append(ql_row['Mean Absolute Error'].iloc[0])
            if 'Mean Time (s)' in ql_row.columns:
                quantlib_times.append(ql_row['Mean Time (s)'].iloc[0])
        else:
            quantlib_errors.append(None)
            quantlib_times.append(None)
    
    # Handle case where theoretical data might be missing
    if not theoretical_data.empty:
        theoretical_mean = theoretical_data['Mean'].iloc[0]
    else:
        # Calculate theoretical mean from parameters if not in results_df
        # Using the parameters from the comparison study
        S0, mu, T = 100, 0.05, 1.0
        theoretical_mean = S0 * np.exp(mu * T)
    
    return samplers, qmcpy_errors, qmcpy_times, quantlib_errors, quantlib_times, theoretical_mean

def add_theoretical_row(results, series_name, n_steps, n_paths, theoretical_mean, theoretical_std):
    """Add theoretical benchmark row to results"""
    results.append({
        'Series': series_name,
        'n_steps': n_steps,
        'n_paths': n_paths,
        'Method': 'Theoretical',
        'Sampler': '-',
        'Mean': theoretical_mean,
        'Std Dev': theoretical_std,
        'Mean Absolute Error': 0,
        'Std Dev Error': 0,
        'Runtime (s)': 0,
        'Runtime Std (s)': 0
    })

def collect_library_results(sampler, series_name, n_steps, n_paths, 
                          ql_timing, qp_timing, theoretical_mean, theoretical_std):
    """Collect results for both QuantLib and QMCPy for a given sampler"""
    results = []
    gbm_params = cf.get_gbm_parameters()
    
    # QuantLib parameters
    ql_params = {**gbm_params, 'n_steps': n_steps, 'n_paths': n_paths}
    
    # QMCPy parameters (note: diffusion = sigma^2)
    qp_params = {
        'initial_value': gbm_params['initial_value'],
        'mu': gbm_params['mu'], 
        'diffusion': gbm_params['sigma']**2,  # Convert sigma to diffusion
        'maturity': gbm_params['maturity'],
        'n_steps': n_steps, 
        'n_paths': n_paths
    }
    
    # QuantLib results (if supported)
    if sampler in cf.get_sampler_configurations()['quantlib_samplers']:
        try:
            ql_paths, _ = qlu.generate_quantlib_paths(sampler_type=sampler, **ql_params)
            ql_final = ql_paths[:, -1]
            ql_mean = np.mean(ql_final)
            ql_std = np.std(ql_final, ddof=1)
            
            results.append({
                'Series': series_name,
                'n_steps': n_steps,
                'n_paths': n_paths,
                'Method': 'QuantLib',
                'Sampler': sampler,
                'Mean': ql_mean,
                'Std Dev': ql_std,
                'Mean Absolute Error': abs(ql_mean - theoretical_mean),
                'Std Dev Error': abs(ql_std - theoretical_std),
                'Runtime (s)': ql_timing[sampler]['average'],
                'Runtime Std (s)': ql_timing[sampler]['stdev']
            })
        except Exception as e:
            print(f"      QuantLib {sampler} failed: {e}")
    
    # QMCPy results
    try:
        qp_paths, _ = qpu.generate_qmcpy_paths(sampler_type=sampler, **qp_params)
        qp_final = qp_paths[:, -1]
        qp_mean = np.mean(qp_final)
        qp_std = np.std(qp_final, ddof=1)
        
        results.append({
            'Series': series_name,
            'n_steps': n_steps,
            'n_paths': n_paths,
            'Method': 'QMCPy',
            'Sampler': sampler,
            'Mean': qp_mean,
            'Std Dev': qp_std,
            'Mean Absolute Error': abs(qp_mean - theoretical_mean),
            'Std Dev Error': abs(qp_std - theoretical_std),
            'Runtime (s)': qp_timing[sampler]['average'],
            'Runtime Std (s)': qp_timing[sampler]['stdev']
        })
    except Exception as e:
        print(f"      QMCPy {sampler} failed: {e}")
    
    return results

