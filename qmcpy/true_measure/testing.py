from .geometric_brownian_motion_copy import  GeometricBrownianMotion
from ..discrete_distribution import DigitalNetB2
# Define a sampler (e.g., a low-discrepancy sequence)
sampler = DigitalNetB2(dimension=4, seed=42)

# Initialize a GeometricBrownianMotion instance
gbm = GeometricBrownianMotion(
    sampler=sampler,
    t_final=2,           # Total time for the process
    initial_value=1,     # Starting value of the process
    drift=0.1,           # Drift coefficient
    diffusion=0.2,       # Diffusion coefficient (volatility)
    decomp_type='PCA'    # Decomposition type for covariance
)

# Generate samples
samples = gbm.gen_samples(5)  # Generate 5 sample paths
print("Generated Samples:")
print(samples)

# Print the GeometricBrownianMotion instance for inspection
print("\nGeometricBrownianMotion Summary:")
print(gbm)