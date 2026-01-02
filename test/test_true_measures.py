from qmcpy import *
from qmcpy.util import *
import numpy as np
import scipy.stats
import unittest


class TestTrueMeasure(unittest.TestCase):
    """ General tests for TrueMeasures """
    
    def test_abstract_methods(self):
        d = 2
        tms = [
            Uniform(DigitalNetB2(d, seed=7)),
            Uniform(DigitalNetB2(d, seed=7),lower_bound=[1,2],upper_bound=[2,3]),
            Kumaraswamy(DigitalNetB2(d, seed=7)),
            Kumaraswamy(DigitalNetB2(d, seed=7),a=[2,4],b=[1,3]),
            JohnsonsSU(DigitalNetB2(d, seed=7)),
            JohnsonsSU(DigitalNetB2(d, seed=7),gamma=[1,2],xi=[4,5],delta=[7,8],lam=[10,11]),
            Gaussian(DigitalNetB2(d, seed=7)),
            Gaussian(DigitalNetB2(d, seed=7),mean=[1,2],covariance=[[9,5],[5,9]],decomp_type='Cholesky'),
            Gaussian(Kumaraswamy(Kumaraswamy(DigitalNetB2(d, seed=7)))),
            BrownianMotion(DigitalNetB2(d, seed=7)),
            BrownianMotion(DigitalNetB2(d, seed=7),t_final=2,drift=3,decomp_type='Cholesky'),
            BernoulliCont(DigitalNetB2(d, seed=7)),
            BernoulliCont(DigitalNetB2(d, seed=7),lam=[.25,.75]),
            SciPyWrapper(DigitalNetB2(2, seed=7),[scipy.stats.triang(c=.1),scipy.stats.uniform(loc=1,scale=2)]),
            SciPyWrapper(DigitalNetB2(2, seed=7),scipy.stats.triang(c=.1,loc=1,scale=2)),
        ]
        for tm in tms:
            for _tm in [tm]+tm.spawn(1):
                t = _tm.gen_samples(4)
                self.assertTrue(t.shape==(4,2))
                self.assertTrue(t.dtype==np.float64)
                x = _tm.discrete_distrib.gen_samples(4)
                xtf,jtf = _tm._jacobian_transform_r(x,return_weights=True)
                self.assertTrue(xtf.shape==(4,d),jtf.shape==(4,))
                w = _tm._weight(x)
                self.assertTrue(w.shape==(4,))
                s = str(_tm)
            
    def test_spawn(self):
        d = 3
        tms = [
            Uniform(DigitalNetB2(d, seed=7)),
            Lebesgue(Uniform(DigitalNetB2(d, seed=7))),
            Lebesgue(Gaussian(DigitalNetB2(d, seed=7))),
            Kumaraswamy(DigitalNetB2(d, seed=7)),
            JohnsonsSU(DigitalNetB2(d, seed=7)),
            Gaussian(DigitalNetB2(d, seed=7)),
            Gaussian(Kumaraswamy(Kumaraswamy(DigitalNetB2(d, seed=7)))),
            BrownianMotion(DigitalNetB2(d, seed=7)),
            BernoulliCont(DigitalNetB2(d, seed=7)),
            SciPyWrapper(DigitalNetB2(2, seed=7),scipy.stats.triang(c=.1,loc=1,scale=2)),
        ]
        for tm in tms:
            s = 3
            for spawn_dim in [4,[1,4,6]]:
                spawns = tm.spawn(s=s,dimensions=spawn_dim)
                self.assertTrue(len(spawns)==s)
                self.assertTrue(all(type(spawn)==type(tm) for spawn in spawns))
                self.assertTrue((np.array([spawn.d for spawn in spawns])==spawn_dim).all())
                self.assertTrue((np.array([spawn.transform.d for spawn in spawns])==spawn_dim).all())
                self.assertTrue((np.array([spawn.transform.transform.d for spawn in spawns])==spawn_dim).all())
                self.assertTrue((np.array([spawn.discrete_distrib.d for spawn in spawns])==spawn_dim).all())
                self.assertTrue((all(spawn.discrete_distrib!=tm.discrete_distrib for spawn in spawns)))
                self.assertTrue(all(spawn.transform!=tm.transform for spawn in spawns))

class TestMatern(unittest.TestCase):
    def test_sklearn_equivalence(self):
        points = np.array([[5, 4], [1, 2], [0, 0]])
        mean = np.full(3, 1.1)
        
        m2 = MaternGP(Lattice(dimension = 3,seed=7), points, length_scale = 4, nu = 2.5, variance = 0.01, mean=mean, nugget=1e-6)
        from sklearn import gaussian_process as gp  #checking against scikit's Matern
        kernel2 = gp.kernels.Matern(length_scale = 4, nu=2.5)
        cov2 = 0.01 * kernel2.__call__(points) + 1e-6*np.eye(m2.covariance.shape[-1])
        assert np.allclose(cov2, m2.covariance)


class TestGaussian(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with fixed seeds for reproducibility."""
        self.seed = 42
    
    def test_gaussian_basic_output_reproducibility(self):
        """Test that basic Gaussian sample generation produces expected values with fixed seed."""
        gaussian = Gaussian(
            Lattice(4, seed=self.seed),
            mean=0,
            covariance=1
        )
        
        samples = gaussian.gen_samples(2)
        
        # Expected output based on fixed seed
        expected_samples = np.array([
            [0.0751091, -0.3100827, 1.60190625, -0.19901344],
            [-1.88173437, 1.16627832, -0.13726206, 1.41268702]
        ])
        
        np.testing.assert_array_almost_equal(
            samples, expected_samples, decimal=6,
            err_msg="Gaussian sample generation output changed unexpectedly"
        )
    
    def test_gaussian_transformation_consistency(self):
        """Test that Gaussian transformation is mathematically consistent."""
        n_paths = 4
        d = 8
        sampler = Lattice(d, seed=self.seed)
        
        # Create Gaussian measure 
        gaussian = Gaussian(sampler, mean=0, covariance=1)
        
        # Generate uniform samples
        uniform_samples = sampler.gen_samples(n_paths)
        
        # Manual original transformation
        from scipy.stats import norm
        normal_samples = norm.ppf(uniform_samples)
        original_result = gaussian.mu + np.einsum("...ij,kj->...ik", normal_samples, gaussian.a)
        
        # Optimized transformation
        optimized_result = gaussian._transform(uniform_samples)
        
        # Check consistency
        np.testing.assert_array_almost_equal(
            original_result, optimized_result, decimal=10,
            err_msg="Gaussian transformation methods are inconsistent"
        )
    
    def test_gaussian_custom_mean_covariance(self):
        """Test Gaussian with custom mean and covariance parameters."""
        custom_mean = np.array([1.0, 2.0])
        custom_cov = np.array([[4.0, 1.0], [1.0, 3.0]])
        
        gaussian = Gaussian(
            Lattice(2, seed=self.seed),
            mean=custom_mean,
            covariance=custom_cov
        )
        
        samples = gaussian.gen_samples(2)
        
        # Expected output with custom parameters
        expected_samples = np.array([
            [0.88570154, 2.49195225],
            [-1.49352915, -1.65710039]
        ])
        
        np.testing.assert_array_almost_equal(
            samples, expected_samples, decimal=6,
            err_msg="Gaussian with custom parameters output changed unexpectedly"
        )
    
    def test_gaussian_weight_computation(self):
        """Test that Gaussian PDF weight computation produces expected values."""
        gaussian = Gaussian(
            Lattice(2, seed=self.seed),
            mean=np.array([0.0, 0.0]),
            covariance=np.eye(2)
        )
        
        # Test with specific known samples
        test_samples = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [-1.0, 1.0]
        ])
        
        weights = gaussian._weight(test_samples)
        
        # Expected weights for standard bivariate normal
        expected_weights = np.array([0.15915494, 0.05854983, 0.05854983])
        
        np.testing.assert_array_almost_equal(
            weights, expected_weights, decimal=6,
            err_msg="Gaussian weight computation changed unexpectedly"
        )
    
    def test_gaussian_decomposition_types(self):
        """Test different decomposition types for Gaussian."""
        custom_cov = np.array([[4.0, 2.0], [2.0, 3.0]])
        
        # Test Cholesky decomposition
        gaussian_chol = Gaussian(
            Lattice(2, seed=self.seed),
            mean=0,
            covariance=custom_cov,
            decomp_type='Cholesky'
        )
        
        # Test PCA decomposition  
        gaussian_pca = Gaussian(
            Lattice(2, seed=self.seed),
            mean=0,
            covariance=custom_cov,
            decomp_type='PCA'
        )
        
        samples_chol = gaussian_chol.gen_samples(2)
        samples_pca = gaussian_pca.gen_samples(2)
        
        # Both should produce valid samples (different due to decomposition method)
        self.assertEqual(samples_chol.shape, (2, 2))
        self.assertEqual(samples_pca.shape, (2, 2))
        self.assertEqual(samples_chol.dtype, np.float64)
        self.assertEqual(samples_pca.dtype, np.float64)
    
    def test_gaussian_mean_covariance_properties(self):
        """Test that Gaussian maintains correct mean and covariance properties."""
        custom_mean = np.array([1.0, -1.0, 2.0])
        custom_cov = np.array([
            [2.0, 0.5, 0.0],
            [0.5, 1.5, -0.3],
            [0.0, -0.3, 3.0]
        ])
        
        gaussian = Gaussian(
            Lattice(3, seed=self.seed),
            mean=custom_mean,
            covariance=custom_cov
        )
        
        # Verify mean is stored correctly
        np.testing.assert_array_almost_equal(
            gaussian.mu, custom_mean, decimal=10,
            err_msg="Gaussian mean property changed unexpectedly"
        )
        
        # Verify covariance reconstruction
        reconstructed_cov = gaussian.a @ gaussian.a.T
        np.testing.assert_array_almost_equal(
            reconstructed_cov, custom_cov, decimal=10,
            err_msg="Gaussian covariance reconstruction changed unexpectedly"
        )
    
    def test_gaussian_scalar_parameters(self):
        """Test Gaussian with scalar mean and covariance parameters."""
        gaussian = Gaussian(
            Lattice(3, seed=self.seed),
            mean=2.5,
            covariance=1.5
        )
        
        samples = gaussian.gen_samples(2)
        
        # Expected samples with scalar parameters
        expected_samples = np.array([
            [2.59198948, 2.1202278, 4.46192646],
            [0.19535548, 3.92839339, 2.33188899]
        ])
        
        np.testing.assert_array_almost_equal(
            samples, expected_samples, decimal=6,
            err_msg="Gaussian with scalar parameters output changed unexpectedly"
        )
        
class TestBrownianMotion(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with fixed seeds for reproducibility."""
        self.seed = 7
    
    def test_brownian_motion_parent_values(self):
        """Test that underlying Brownian Motion values are correct."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=1,
            drift=0.1,
            diffusion=0.2
        )
        
        # Test time vector from parent
        expected_time_vec = np.array([0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(
            gbm.time_vec, expected_time_vec, decimal=10,
            err_msg="Time vector from parent BrownianMotion changed unexpectedly"
        )
        
        # Test parent's covariance matrix 
        expected_parent_cov = gbm.diffusion * np.minimum.outer(gbm.time_vec, gbm.time_vec)

        # Access parent's covariance 
        if hasattr(gbm, 'covariance'):
            np.testing.assert_array_almost_equal(
                gbm.covariance, expected_parent_cov, decimal=10,
                err_msg="Parent BrownianMotion covariance changed unexpectedly"
            )


class TestGeometricBrownianMotion(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with fixed seeds for reproducibility."""
        self.seed = 7
    
    def test_gbm_basic_output_reproducibility(self):
        """Test that basic GBM sample generation produces expected values with fixed seed."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed), 
            t_final=2, 
            drift=0.1, 
            diffusion=0.2
        )
        
        samples = gbm.gen_samples(2)
        
        # Expected output 
        expected_samples = np.array([
            [0.92343761, 1.42069027, 1.30851806, 0.99133819],
            [0.7185916, 0.42028013, 0.42080335, 0.4696196]
        ])
        
        np.testing.assert_array_almost_equal(
            samples, expected_samples, decimal=6,
            err_msg="GBM sample generation output changed unexpectedly"
        )
        
    def test_gbm_mean_computation(self):
        """Test that GBM mean computation produces expected values."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=2,
            initial_value=1,
            drift=0.1,
            diffusion=0.2
        )
        
        expected_mean = np.array([1.051271096, 1.105170918, 1.161834243, 1.221402758])
        
        np.testing.assert_array_almost_equal(
            gbm.mean_gbm, expected_mean, decimal=6,
            err_msg="GBM mean computation changed unexpectedly"
        )
        
    def test_gbm_covariance_computation(self):
        """Test that GBM covariance computation produces expected values."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=2,
            initial_value=1,
            drift=0.1,
            diffusion=0.2
        )
        
        # Expected covariance matrix - actual computed values
        expected_cov = np.array([
            [0.116232, 0.122191, 0.128456, 0.135042],
            [0.122191, 0.270422, 0.284287, 0.298862],
            [0.128456, 0.284287, 0.47226, 0.496473],
            [0.135042, 0.298862, 0.496473, 0.733716]
        ])
        
        np.testing.assert_array_almost_equal(
            gbm.covariance_gbm, expected_cov, decimal=6,
            err_msg="GBM covariance computation changed unexpectedly"
        )

    def test_gbm_weight_specific_values(self):
        """Test that PDF weight computation produces expected values for specific inputs."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=1,
            initial_value=1,
            drift=0.05,
            diffusion=0.1
        )
        
        # Test with specific known samples
        test_samples = np.array([
            [1.0, 1.05, 1.1, 1.15],
            [0.9, 0.95, 1.0, 1.05]
        ])
        
        weights = gbm._weight(test_samples)
        
        # Expected weights - actual computed values
        expected_weights = np.array([26.782039, 30.850616])
        
        np.testing.assert_array_almost_equal(
            weights, expected_weights, decimal=6,
            err_msg="GBM weight computation changed unexpectedly"
        )

    def test_transform_specific_inputs(self):
            """Test _transform method with specific uniform inputs."""
            gbm = GeometricBrownianMotion(
                DigitalNetB2(4, seed=self.seed),
                t_final=1,
                drift=0.1,
                diffusion=0.2,
                initial_value=2.0
            )
            
            # Specific uniform inputs
            uniform_inputs = np.array([
                [0.1, 0.3, 0.7, 0.9],
                [0.5, 0.5, 0.5, 0.5]
            ])
            
            transformed = gbm._transform(uniform_inputs)
            
            # Expected transformed values - actual computed values
            expected_transformed = np.array([
                [1.73828, 1.166842, 1.297715, 1.24253],
                [2.0, 2.0, 2.0, 2.0]
            ])
            
            np.testing.assert_array_almost_equal(
                transformed, expected_transformed, decimal=6,
                err_msg="Transform method output changed unexpectedly"
            )

    def test_setup_lognormal_distribution_properties(self):
        """Test that _setup_lognormal_distribution creates correct log-space distribution."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=2,
            initial_value=1,
            drift=0.1,
            diffusion=0.2
        )
        
        # Access log_mvn_scipy to trigger setup
        log_mvn = gbm.log_mvn_scipy
        
        # Test log-space mean: (drift - 0.5*diffusion) * time_vec
        expected_log_mean = (0.1 - 0.5 * 0.2) * gbm.time_vec
        np.testing.assert_array_almost_equal(
            log_mvn.mean, expected_log_mean, decimal=10,
            err_msg="Log-space mean computation in lognormal setup changed unexpectedly"
        )
        
        # Test log-space covariance structure: diffusion * min(t_i, t_j)
        expected_log_cov = 0.2 * np.minimum.outer(gbm.time_vec, gbm.time_vec)
        np.testing.assert_array_almost_equal(
            log_mvn.cov, expected_log_cov, decimal=10,
            err_msg="Log-space covariance computation in lognormal setup changed unexpectedly"
        )

    def test_lognormal_distribution_pdf_consistency(self):
        """Test that the lognormal distribution PDF is mathematically consistent."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=1,
            initial_value=2,
            drift=0.05,
            diffusion=0.1
        )
        
        # Test with a known sample
        test_sample = np.array([[2.1, 2.15, 2.2, 2.25]])
        gbm_weight = gbm._weight(test_sample)[0]
        
        # Manually compute using log transformation
        log_sample = np.log(test_sample / gbm.initial_value).flatten()
        normal_pdf = gbm.log_mvn_scipy.pdf(log_sample)
        jacobian = 1.0 / test_sample.prod()
        manual_weight = normal_pdf * jacobian
        
        np.testing.assert_almost_equal(
            gbm_weight, manual_weight, decimal=10,
            err_msg="Lognormal PDF computation inconsistency detected"
        )

    def test_lognormal_setup_lazy_loading(self):
        """Test that lognormal distribution setup works with lazy loading."""
        # Test with lazy_load=True (default)
        gbm_lazy = GeometricBrownianMotion(
            DigitalNetB2(3, seed=self.seed),
            t_final=1,
            drift=0.1,
            diffusion=0.2,
            lazy_load=True
        )
        self.assertIsNone(gbm_lazy._log_mvn_scipy_cache)
        log_mvn = gbm_lazy.log_mvn_scipy
        self.assertIsNotNone(gbm_lazy._log_mvn_scipy_cache)
        
        # Test with lazy_load=False
        gbm_eager = GeometricBrownianMotion(
            DigitalNetB2(3, seed=self.seed),
            t_final=1,
            drift=0.1,
            diffusion=0.2,
            lazy_load=False
        )
        self.assertIsNotNone(gbm_eager._log_mvn_scipy_cache)
    
if __name__ == "__main__":
    unittest.main()
