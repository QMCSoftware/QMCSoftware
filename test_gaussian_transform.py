#!/usr/bin/env python3
"""
Compare original vs optimized Gaussian transformation.
"""

import numpy as np
import qmcpy as qp
from scipy.stats import norm

def test_gaussian_transformation():
    """Compare original vs optimized Gaussian transformation"""
    print("Testing Gaussian transformation...")
    
    # Create a simple Gaussian measure to test transformation
    n_paths = 4
    d = 128
    sampler = qp.Lattice(d, seed=42)
    
    # Create Gaussian measure 
    gaussian = qp.Gaussian(sampler, mean=0, covariance=1)
    
    # Generate uniform samples
    uniform_samples = sampler.gen_samples(n_paths)
    print(f"Uniform samples shape: {uniform_samples.shape}")
    print(f"Uniform min/max: {np.min(uniform_samples):.6f} / {np.max(uniform_samples):.6f}")
    
    # Manual original transformation
    print("\n--- Original method ---")
    normal_samples = norm.ppf(uniform_samples)
    original_result = gaussian.mu + np.einsum("...ij,kj->...ik", normal_samples, gaussian.a)
    print(f"Original result shape: {original_result.shape}")
    print(f"Original min/max: {np.min(original_result):.6f} / {np.max(original_result):.6f}")
    
    # Our optimized transformation
    print("\n--- Optimized method ---")
    optimized_result = gaussian._transform(uniform_samples)
    print(f"Optimized result shape: {optimized_result.shape}")
    print(f"Optimized min/max: {np.min(optimized_result):.6f} / {np.max(optimized_result):.6f}")
    
    # Check differences
    print("\n--- Comparison ---")
    max_diff = np.max(np.abs(original_result - optimized_result))
    print(f"Max difference: {max_diff:.10f}")
    
    if max_diff > 1e-10:
        print("WARNING: Significant differences found!")
        diff_mask = np.abs(original_result - optimized_result) > 1e-10
        diff_positions = np.where(diff_mask)
        for i in range(min(5, len(diff_positions[0]))):
            row, col = diff_positions[0][i], diff_positions[1][i]
            print(f"  Position ({row}, {col}): original={original_result[row, col]:.6f}, optimized={optimized_result[row, col]:.6f}")
    
    return original_result, optimized_result

if __name__ == "__main__":
    test_gaussian_transformation()
