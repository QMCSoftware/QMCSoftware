from qmcpy import *
from numpy import *
import unittest


class TestGeometricBrownianMotion(unittest.TestCase):

    def test_sample_moments(self):
        # Parameters
        S0, mu, sigma, T, n_samples = 100.0, 0.05, 0.20, 1.0, 2**12
        sampler = IIDStdUniform(5, seed=42)
        gbm = GeometricBrownianMotion(sampler, t_final=T, initial_value=S0, drift=mu, diffusion=sigma)
        paths = gbm.gen_samples(n_samples)
        S_T = paths[:, -1]

        # Theoretical moments
        theo_mean = S0 * exp(mu * T)
        theo_var = S0**2 * exp(2*mu*T) * (exp(sigma**2 * T) - 1)

        # Empirical moments
        emp_mean = mean(S_T)
        emp_var = var(S_T, ddof=1)
        
        print(f"Mean: {emp_mean:.2f} (theoretical: {theo_mean:.2f})")
        print(f"Variance: {emp_var:.2f} (theoretical: {theo_var:.2f})")
        self.assertTrue(abs(emp_mean - theo_mean) / theo_mean < 0.1)
        self.assertTrue(abs(emp_var - theo_var) / theo_var < 0.1)