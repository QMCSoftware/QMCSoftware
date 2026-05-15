from qmcpy import *
from qmcpy.util import *
import numpy as np
import scipy.stats
import unittest
from qmcpy.true_measure.uniform_triangle import UniformTriangle, _UniformTriangleAdapter
from qmcpy.true_measure.scipy_wrapper import SciPyWrapper


class TestTrueMeasure(unittest.TestCase):
    """General tests for TrueMeasures"""

    def test_abstract_methods(self):
        d = 2
        tms = [
            Uniform(DigitalNetB2(d, seed=7)),
            Uniform(DigitalNetB2(d, seed=7), lower_bound=[1, 2], upper_bound=[2, 3]),
            Kumaraswamy(DigitalNetB2(d, seed=7)),
            Kumaraswamy(DigitalNetB2(d, seed=7), a=[2, 4], b=[1, 3]),
            JohnsonsSU(DigitalNetB2(d, seed=7)),
            JohnsonsSU(
                DigitalNetB2(d, seed=7),
                gamma=[1, 2],
                xi=[4, 5],
                delta=[7, 8],
                lam=[10, 11],
            ),
            Gaussian(DigitalNetB2(d, seed=7)),
            Gaussian(
                DigitalNetB2(d, seed=7),
                mean=[1, 2],
                covariance=[[9, 5], [5, 9]],
                decomp_type="Cholesky",
            ),
            Gaussian(Kumaraswamy(Kumaraswamy(DigitalNetB2(d, seed=7)))),
            BrownianMotion(DigitalNetB2(d, seed=7)),
            BrownianMotion(
                DigitalNetB2(d, seed=7), t_final=2, drift=3, decomp_type="Cholesky"
            ),
            BernoulliCont(DigitalNetB2(d, seed=7)),
            BernoulliCont(DigitalNetB2(d, seed=7), lam=[0.25, 0.75]),
            SciPyWrapper(
                DigitalNetB2(2, seed=7),
                [scipy.stats.triang(c=0.1), scipy.stats.uniform(loc=1, scale=2)],
            ),
            SciPyWrapper(
                DigitalNetB2(2, seed=7), scipy.stats.triang(c=0.1, loc=1, scale=2)
            ),
        ]
        for tm in tms:
            for _tm in [tm] + tm.spawn(1):
                t = _tm.gen_samples(4)
                self.assertEqual(t.shape, (4, 2))
                self.assertEqual(t.dtype, np.float64)
                x = _tm.discrete_distrib.gen_samples(4)
                xtf, jtf = _tm._jacobian_transform_r(x, return_weights=True)
                self.assertTrue(xtf.shape == (4, d), jtf.shape == (4,))
                w = _tm._weight(x)
                self.assertEqual(w.shape, (4,))
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
            SciPyWrapper(
                DigitalNetB2(2, seed=7), scipy.stats.triang(c=0.1, loc=1, scale=2)
            ),
        ]
        for tm in tms:
            s = 3
            for spawn_dim in [4, [1, 4, 6]]:
                spawns = tm.spawn(s=s, dimensions=spawn_dim)
                self.assertEqual(len(spawns), s)
                self.assertTrue(all(type(spawn) == type(tm) for spawn in spawns))
                self.assertTrue(
                    (np.array([spawn.d for spawn in spawns]) == spawn_dim).all()
                )
                self.assertTrue(
                    (
                        np.array([spawn.transform.d for spawn in spawns]) == spawn_dim
                    ).all()
                )
                self.assertTrue(
                    (
                        np.array([spawn.transform.transform.d for spawn in spawns])
                        == spawn_dim
                    ).all()
                )
                self.assertTrue(
                    (
                        np.array([spawn.discrete_distrib.d for spawn in spawns])
                        == spawn_dim
                    ).all()
                )
                self.assertTrue(
                    (
                        all(
                            spawn.discrete_distrib != tm.discrete_distrib
                            for spawn in spawns
                        )
                    )
                )
                self.assertTrue(
                    all(spawn.transform != tm.transform for spawn in spawns)
                )


class TestMatern(unittest.TestCase):
    def test_sklearn_equivalence(self):
        points = np.array([[5, 4], [1, 2], [0, 0]])
        mean = np.full(3, 1.1)

        m2 = MaternGP(
            Lattice(dimension=3, seed=7),
            points,
            length_scale=4,
            nu=2.5,
            variance=0.01,
            mean=mean,
            nugget=1e-6,
        )
        from sklearn import gaussian_process as gp  # checking against scikit's Matern

        kernel2 = gp.kernels.Matern(length_scale=4, nu=2.5)
        cov2 = 0.01 * kernel2.__call__(points) + 1e-6 * np.eye(m2.covariance.shape[-1])
        assert np.allclose(cov2, m2.covariance)

class TestUniformTriangle(unittest.TestCase):
    """Tests for UniformTriangle and _UniformTriangleAdapter."""

    def test_basic_usage_and_dim_error(self):
        tm = UniformTriangle(sampler=DigitalNetB2(2, seed=7))
        x = tm(8)
        self.assertEqual(x.shape, (8, 2))
        self.assertTrue(np.all(x[:, 1] <= x[:, 0]))

        with self.assertRaises(DimensionError):
            tm._joint.transform(np.ones((4, 3)))

    def test_adapter_transform(self):
        adapter = _UniformTriangleAdapter()
        u = np.array([[0.25, 0.5], [0.64, 0.75], [0.01, 0.9]])
        x = adapter.transform(u)
        self.assertEqual(x.shape, (3, 2))
        # y = u2 * sqrt(u1) <= sqrt(u1) = x  always since u2 in [0,1]
        self.assertTrue(np.all(x[:, 1] <= x[:, 0] + 1e-12))

    def test_adapter_transform_dim_error(self):
        adapter = _UniformTriangleAdapter()
        # Wrong last dimension (3 instead of 2)
        self.assertRaises(DimensionError, adapter.transform, np.ones((4, 3)))

    def test_adapter_logpdf_inside(self):
        adapter = _UniformTriangleAdapter()
        # Points inside the triangle (y <= x, x in [0,1])
        x = np.array([[0.5, 0.3], [0.8, 0.0], [1.0, 0.5]])
        lp = adapter.logpdf(x)
        self.assertEqual(lp.shape, (3,))
        self.assertTrue(np.all(np.isfinite(lp)))

    def test_adapter_logpdf_outside(self):
        adapter = _UniformTriangleAdapter()
        # Points outside the triangle (y > x)
        x = np.array([[0.3, 0.5]])
        lp = adapter.logpdf(x)
        self.assertEqual(lp[0], -np.inf)

    def test_adapter_logpdf_dim_error(self):
        adapter = _UniformTriangleAdapter()
        self.assertRaises(DimensionError, adapter.logpdf, np.ones((4, 3)))

    def test_adapter_logpdf_batched(self):
        adapter = _UniformTriangleAdapter()
        u = np.random.default_rng(42).uniform(size=(5, 2))
        x = adapter.transform(u)
        lp = adapter.logpdf(x)
        self.assertEqual(lp.shape, (5,))


class TestGaussian(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with fixed seeds for reproducibility."""
        self.seed = 42

    def test_gaussian_basic_output_reproducibility(self):
        """Test that basic Gaussian sample generation produces expected values with fixed seed."""
        gaussian = Gaussian(Lattice(4, seed=self.seed), mean=0, covariance=1)

        samples = gaussian.gen_samples(2)

        # Expected output based on fixed seed
        expected_samples = np.array(
            [
                [0.0751091, -0.3100827, 1.60190625, -0.19901344],
                [-1.88173437, 1.16627832, -0.13726206, 1.41268702],
            ]
        )

        np.testing.assert_array_almost_equal(
            samples,
            expected_samples,
            decimal=6,
            err_msg="Gaussian sample generation output changed unexpectedly",
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
        original_result = gaussian.mu + np.einsum(
            "...ij,kj->...ik", normal_samples, gaussian.a
        )

        # Optimized transformation
        optimized_result = gaussian._transform(uniform_samples)

        # Check consistency
        np.testing.assert_array_almost_equal(
            original_result,
            optimized_result,
            decimal=10,
            err_msg="Gaussian transformation methods are inconsistent",
        )

    def test_gaussian_custom_mean_covariance(self):
        """Test Gaussian with custom mean and covariance parameters."""
        custom_mean = np.array([1.0, 2.0])
        custom_cov = np.array([[4.0, 1.0], [1.0, 3.0]])

        gaussian = Gaussian(
            Lattice(2, seed=self.seed), mean=custom_mean, covariance=custom_cov
        )

        samples = gaussian.gen_samples(2)

        # Expected output with custom parameters
        expected_samples = np.array(
            [[0.88570154, 2.49195225], [-1.49352915, -1.65710039]]
        )

        np.testing.assert_array_almost_equal(
            samples,
            expected_samples,
            decimal=6,
            err_msg="Gaussian with custom parameters output changed unexpectedly",
        )

    def test_gaussian_weight_computation(self):
        """Test that Gaussian PDF weight computation produces expected values."""
        gaussian = Gaussian(
            Lattice(2, seed=self.seed), mean=np.array([0.0, 0.0]), covariance=np.eye(2)
        )

        # Test with specific known samples
        test_samples = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])

        weights = gaussian._weight(test_samples)

        # Expected weights for standard bivariate normal
        expected_weights = np.array([0.15915494, 0.05854983, 0.05854983])

        np.testing.assert_array_almost_equal(
            weights,
            expected_weights,
            decimal=6,
            err_msg="Gaussian weight computation changed unexpectedly",
        )

    def test_gaussian_decomposition_types(self):
        """Test different decomposition types for Gaussian."""
        custom_cov = np.array([[4.0, 2.0], [2.0, 3.0]])

        # Test Cholesky decomposition
        gaussian_chol = Gaussian(
            Lattice(2, seed=self.seed),
            mean=0,
            covariance=custom_cov,
            decomp_type="Cholesky",
        )

        # Test PCA decomposition
        gaussian_pca = Gaussian(
            Lattice(2, seed=self.seed), mean=0, covariance=custom_cov, decomp_type="PCA"
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
        custom_cov = np.array([[2.0, 0.5, 0.0], [0.5, 1.5, -0.3], [0.0, -0.3, 3.0]])

        gaussian = Gaussian(
            Lattice(3, seed=self.seed), mean=custom_mean, covariance=custom_cov
        )

        # Verify mean is stored correctly
        np.testing.assert_array_almost_equal(
            gaussian.mu,
            custom_mean,
            decimal=10,
            err_msg="Gaussian mean property changed unexpectedly",
        )

        # Verify covariance reconstruction
        reconstructed_cov = gaussian.a @ gaussian.a.T
        np.testing.assert_array_almost_equal(
            reconstructed_cov,
            custom_cov,
            decimal=10,
            err_msg="Gaussian covariance reconstruction changed unexpectedly",
        )

    def test_gaussian_scalar_parameters(self):
        """Test Gaussian with scalar mean and covariance parameters."""
        gaussian = Gaussian(Lattice(3, seed=self.seed), mean=2.5, covariance=1.5)

        samples = gaussian.gen_samples(2)

        # Expected samples with scalar parameters
        expected_samples = np.array(
            [[2.59198948, 2.1202278, 4.46192646], [0.19535548, 3.92839339, 2.33188899]]
        )

        np.testing.assert_array_almost_equal(
            samples,
            expected_samples,
            decimal=6,
            err_msg="Gaussian with scalar parameters output changed unexpectedly",
        )


class TestBrownianMotion(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with fixed seeds for reproducibility."""
        self.seed = 7

    def test_brownian_motion_parent_values(self):
        """Test that underlying Brownian Motion values are correct."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed), t_final=1, drift=0.1, diffusion=0.2
        )

        # Test time vector from parent
        expected_time_vec = np.array([0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(
            gbm.time_vec,
            expected_time_vec,
            decimal=10,
            err_msg="Time vector from parent BrownianMotion changed unexpectedly",
        )

        # Test parent's covariance matrix
        expected_parent_cov = gbm.diffusion * np.minimum.outer(
            gbm.time_vec, gbm.time_vec
        )

        # Access parent's covariance
        if hasattr(gbm, "covariance"):
            np.testing.assert_array_almost_equal(
                gbm.covariance,
                expected_parent_cov,
                decimal=10,
                err_msg="Parent BrownianMotion covariance changed unexpectedly",
            )


class TestGeometricBrownianMotion(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with fixed seeds for reproducibility."""
        self.seed = 7

    def test_gbm_basic_output_reproducibility(self):
        """Test that basic GBM sample generation produces expected values with fixed seed."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed), t_final=2, drift=0.1, diffusion=0.2
        )

        samples = gbm.gen_samples(2)

        # Expected output
        expected_samples = np.array(
            [
                [0.92343761, 1.42069027, 1.30851806, 0.99133819],
                [0.7185916, 0.42028013, 0.42080335, 0.4696196],
            ]
        )

        np.testing.assert_array_almost_equal(
            samples,
            expected_samples,
            decimal=6,
            err_msg="GBM sample generation output changed unexpectedly",
        )

    def test_gbm_mean_computation(self):
        """Test that GBM mean computation produces expected values."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=2,
            initial_value=1,
            drift=0.1,
            diffusion=0.2,
        )

        expected_mean = np.array([1.051271096, 1.105170918, 1.161834243, 1.221402758])

        np.testing.assert_array_almost_equal(
            gbm.mean_gbm,
            expected_mean,
            decimal=6,
            err_msg="GBM mean computation changed unexpectedly",
        )

    def test_gbm_covariance_computation(self):
        """Test that GBM covariance computation produces expected values."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=2,
            initial_value=1,
            drift=0.1,
            diffusion=0.2,
        )

        # Expected covariance matrix - actual computed values
        expected_cov = np.array(
            [
                [0.116232, 0.122191, 0.128456, 0.135042],
                [0.122191, 0.270422, 0.284287, 0.298862],
                [0.128456, 0.284287, 0.47226, 0.496473],
                [0.135042, 0.298862, 0.496473, 0.733716],
            ]
        )

        np.testing.assert_array_almost_equal(
            gbm.covariance_gbm,
            expected_cov,
            decimal=6,
            err_msg="GBM covariance computation changed unexpectedly",
        )

    def test_gbm_weight_specific_values(self):
        """Test that PDF weight computation produces expected values for specific inputs."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=1,
            initial_value=1,
            drift=0.05,
            diffusion=0.1,
        )

        # Test with specific known samples
        test_samples = np.array([[1.0, 1.05, 1.1, 1.15], [0.9, 0.95, 1.0, 1.05]])

        weights = gbm._weight(test_samples)

        # Expected weights - actual computed values
        expected_weights = np.array([26.782039, 30.850616])

        np.testing.assert_array_almost_equal(
            weights,
            expected_weights,
            decimal=6,
            err_msg="GBM weight computation changed unexpectedly",
        )

    def test_transform_specific_inputs(self):
        """Test _transform method with specific uniform inputs."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=1,
            drift=0.1,
            diffusion=0.2,
            initial_value=2.0,
        )

        # Specific uniform inputs
        uniform_inputs = np.array([[0.1, 0.3, 0.7, 0.9], [0.5, 0.5, 0.5, 0.5]])

        transformed = gbm._transform(uniform_inputs)

        # Expected transformed values - actual computed values
        expected_transformed = np.array(
            [[1.73828, 1.166842, 1.297715, 1.24253], [2.0, 2.0, 2.0, 2.0]]
        )

        np.testing.assert_array_almost_equal(
            transformed,
            expected_transformed,
            decimal=6,
            err_msg="Transform method output changed unexpectedly",
        )

    def test_setup_lognormal_distribution_properties(self):
        """Test that _setup_lognormal_distribution creates correct log-space distribution."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=2,
            initial_value=1,
            drift=0.1,
            diffusion=0.2,
        )

        # Access log_mvn_scipy to trigger setup
        log_mvn = gbm.log_mvn_scipy

        # Test log-space mean: (drift - 0.5*diffusion) * time_vec
        expected_log_mean = (0.1 - 0.5 * 0.2) * gbm.time_vec
        np.testing.assert_array_almost_equal(
            log_mvn.mean,
            expected_log_mean,
            decimal=10,
            err_msg="Log-space mean computation in lognormal setup changed unexpectedly",
        )

        # Test log-space covariance structure: diffusion * min(t_i, t_j)
        expected_log_cov = 0.2 * np.minimum.outer(gbm.time_vec, gbm.time_vec)
        np.testing.assert_array_almost_equal(
            log_mvn.cov,
            expected_log_cov,
            decimal=10,
            err_msg="Log-space covariance computation in lognormal setup changed unexpectedly",
        )

    def test_lognormal_distribution_pdf_consistency(self):
        """Test that the lognormal distribution PDF is mathematically consistent."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=1,
            initial_value=2,
            drift=0.05,
            diffusion=0.1,
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
            gbm_weight,
            manual_weight,
            decimal=10,
            err_msg="Lognormal PDF computation inconsistency detected",
        )

    def test_lognormal_setup_lazy_loading(self):
        """Test that lognormal distribution setup works with lazy loading."""
        # Test with lazy_load=True (default)
        gbm_lazy = GeometricBrownianMotion(
            DigitalNetB2(3, seed=self.seed),
            t_final=1,
            drift=0.1,
            diffusion=0.2,
            lazy_load=True,
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
            lazy_load=False,
        )
        self.assertIsNotNone(gbm_eager._log_mvn_scipy_cache)


class TestAcceptanceRejection(unittest.TestCase):
    """Unit tests for AcceptanceRejection and AcceptanceRejectionReal."""

    def setUp(self):
        from qmcpy.true_measure import AcceptanceRejection, AcceptanceRejectionReal
        from scipy.stats import norm
        self.AcceptanceRejection = AcceptanceRejection
        self.AcceptanceRejectionReal = AcceptanceRejectionReal
        self.norm = norm
        self.seed = 7

        def psi(x): return 2 * x[:, 0]
        self.psi = psi

        def psi_real(z): return norm.pdf(z[:, 0], loc=0, scale=1)
        def H(z): return norm.pdf(z[:, 0], loc=0, scale=2)
        self.psi_real = psi_real
        self.H = H
        self.inv_cdfs = [lambda u: norm.ppf(u, loc=0, scale=2)]

    def _make_ar(self, seed=None):
        seed = seed if seed is not None else self.seed
        return self.AcceptanceRejection(
            DigitalNetB2(dimension=2, seed=seed),
            self.psi, upper_bound=2., density_integral=1.
        )

    def _make_ar_real(self, seed=None):
        seed = seed if seed is not None else self.seed
        return self.AcceptanceRejectionReal(
            DigitalNetB2(dimension=2, seed=seed),
            self.psi_real,
            inv_cdfs=self.inv_cdfs,
            H_func=self.H,
            upper_bound=2., density_integral=1.
        )

    # --- AcceptanceRejection tests ---

    def test_basic_shape(self):
        """gen_samples(n=64) returns shape (64, 1)."""
        samples = self._make_ar().gen_samples(n=64, warn=False)
        self.assertEqual(samples.shape, (64, 1))

    def test_samples_in_unit_interval(self):
        """All accepted samples lie in [0, 1]."""
        samples = self._make_ar().gen_samples(n=256, warn=False)
        self.assertTrue((samples >= 0).all() and (samples <= 1).all())

    def test_mean_convergence(self):
        """Sample mean should be close to the true mean of 2/3."""
        samples = self._make_ar().gen_samples(n=1024, warn=False)
        self.assertAlmostEqual(samples.mean(), 2/3, delta=0.05)

    def test_return_weights(self):
        """return_weights=True gives samples and positive weights of correct shape."""
        s, w = self._make_ar().gen_samples(n=64, return_weights=True, warn=False)
        self.assertEqual(s.shape, (64, 1))
        self.assertEqual(w.shape, (64,))
        self.assertTrue(np.all(w > 0))

    def test_continued_sampling_matches_single_call(self):
        """Two batches via n_min/n_max equal one single call."""
        m1 = self._make_ar()
        b1 = m1.gen_samples(n_min=0, n_max=8)
        b2 = m1.gen_samples(n_min=8, n_max=16)

        m2 = self._make_ar()
        all_at_once = m2.gen_samples(n_min=0, n_max=16)

        np.testing.assert_array_almost_equal(
            np.concatenate([b1, b2]), all_at_once
        )

    def test_n_resets_driver(self):
        """Calling gen_samples(n=8) twice gives same result (driver resets)."""
        m = self._make_ar()
        s1 = m.gen_samples(n=8)
        s2 = m.gen_samples(n=8)
        np.testing.assert_array_equal(s1, s2)

    def test_n_max_without_n_min(self):
        """Calling with only n_max (no n_min) works and returns correct shape."""
        s = self._make_ar().gen_samples(n_max=64, warn=False)
        self.assertEqual(s.shape, (64, 1))

    def test_error_n_min_without_prior_call(self):
        """n_min > 0 without a prior call raises ParameterError."""
        m = self._make_ar()
        with self.assertRaises(ParameterError):
            m.gen_samples(n_min=4, n_max=8)

    def test_error_invalid_upper_bound(self):
        """upper_bound <= 0 raises ParameterError."""
        with self.assertRaises(ParameterError):
            self.AcceptanceRejection(
                DigitalNetB2(dimension=2, seed=self.seed),
                self.psi, upper_bound=0., density_integral=1.
            )

    def test_error_invalid_density_integral(self):
        """density_integral <= 0 raises ParameterError."""
        with self.assertRaises(ParameterError):
            self.AcceptanceRejection(
                DigitalNetB2(dimension=2, seed=self.seed),
                self.psi, upper_bound=2., density_integral=0.
            )

    def test_error_sampler_dimension_too_small(self):
        """Sampler dimension < 2 raises ParameterError."""
        with self.assertRaises(ParameterError):
            self.AcceptanceRejection(
                DigitalNetB2(dimension=1, seed=self.seed),
                self.psi, upper_bound=2., density_integral=1.
            )

    # --- AcceptanceRejectionReal tests ---

    def test_real_basic_shape(self):
        """AcceptanceRejectionReal gen_samples(n=64) returns shape (64, 1)."""
        samples = self._make_ar_real().gen_samples(n=64, warn=False)
        self.assertEqual(samples.shape, (64, 1))

    def test_real_mean_and_std(self):
        """Sample mean and std should be close to N(0,1) values."""
        samples = self._make_ar_real().gen_samples(n=512, warn=False)
        self.assertAlmostEqual(samples.mean(), 0.0, delta=0.1)
        self.assertAlmostEqual(samples.std(), 1.0, delta=0.1)

    def test_real_continued_sampling_shapes(self):
        """AcceptanceRejectionReal continued sampling returns correct shapes."""
        m = self._make_ar_real()
        b1 = m.gen_samples(n_min=0, n_max=8)
        b2 = m.gen_samples(n_min=8, n_max=16)
        self.assertEqual(b1.shape, (8, 1))
        self.assertEqual(b2.shape, (8, 1))

    def test_real_error_n_min_without_prior_call(self):
        """AcceptanceRejectionReal n_min > 0 without prior call raises ParameterError."""
        m = self._make_ar_real()
        with self.assertRaises(ParameterError):
            m.gen_samples(n_min=8, n_max=16)

    def test_real_error_inv_cdfs_length_mismatch(self):
        """inv_cdfs length != target_dim raises ParameterError."""
        with self.assertRaises(ParameterError):
            self.AcceptanceRejectionReal(
                DigitalNetB2(dimension=2, seed=self.seed),
                self.psi_real,
                inv_cdfs=[],  # wrong length
                H_func=self.H,
                upper_bound=2., density_integral=1.
            )


if __name__ == "__main__":
    unittest.main()
