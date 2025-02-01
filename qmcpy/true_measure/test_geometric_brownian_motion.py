import numpy as np
from qmcpy.true_measure.geometric_brownian_motion import GeometricBrownianMotion
from qmcpy.discrete_distribution import DigitalNetB2

def test_geometric_brownian_motion():
    """
    Test the GeometricBrownianMotion class.
    """
   
    sampler = DigitalNetB2(4, seed=7)

    gbm = GeometricBrownianMotion(
        sampler=sampler,
        t_final=2,
        initial_value=100,
        drift=0.05,
        diffusion=0.2
    )

    
    print("Geometric Brownian Motion initialized.")
    print("Time vector:", gbm.time_vec)
    print("Mean of GBM:", gbm.mean_gbm)
    print("Covariance of GBM:\n", gbm.covariance_gbm)

    
    num_samples = 2**3
    x = sampler.gen_samples(num_samples)
    transformed_samples = gbm._transform(x)
    print("\nGenerated Samples (transformed):\n", transformed_samples)

    
    assert transformed_samples.shape == (num_samples, len(gbm.time_vec)), \
        "Sample dimensions do not match expected output."

    
    print("\nAll tests passed successfully.")

if __name__ == "__main__":
    
    test_geometric_brownian_motion()