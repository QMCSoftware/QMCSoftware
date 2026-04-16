import unittest
import numpy as np
from qmcpy.util import (
    _univ_repr,
    ParameterError,
    MethodImplementationError,
)
from qmcpy.util.data import Data
from qmcpy.util.torch_numpy_ops import get_npt
from qmcpy.util.transforms import (
    insert_batch_dims,
    tf_exp,
    tf_exp_inv,
    tf_square,
    tf_square_inv,
    tf_identity,
    tf_explinear,
    tf_explinear_inv,
)


class TestTorchNumpyOps(unittest.TestCase):
    """Tests for torch_numpy_ops.get_npt()."""

    def test_get_npt_numpy_array(self):
        """get_npt should return np for numpy arrays."""
        x = np.array([1.0, 2.0, 3.0])
        npt = get_npt(x)
        self.assertEqual(npt, np)

    def test_get_npt_torch_tensor(self):
        """get_npt should return torch for torch tensors."""
        try:
            import torch
            x = torch.tensor([1.0, 2.0, 3.0])
            npt = get_npt(x)
            self.assertEqual(npt, torch)
        except ImportError:
            self.skipTest("torch not installed")


class TestTransforms(unittest.TestCase):
    """Tests for transform functions."""

    def test_insert_batch_dims(self):
        """insert_batch_dims should add batch dimensions."""
        param = np.array([1.0, 2.0, 3.0])
        result = insert_batch_dims(param, ndims=2, k=0)
        self.assertEqual(result.shape, (1, 1, 3))

    def test_insert_batch_dims_middle(self):
        """insert_batch_dims should work with k > 0."""
        param = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = insert_batch_dims(param, ndims=1, k=1)
        self.assertEqual(result.shape, (2, 1, 2))

    def test_tf_exp(self):
        """tf_exp should compute element-wise exponential."""
        x = np.array([0.0, 1.0, 2.0])
        result = tf_exp(x)
        expected = np.exp(x)
        np.testing.assert_allclose(result, expected)

    def test_tf_exp_inv(self):
        """tf_exp_inv should compute element-wise logarithm."""
        x = np.array([1.0, np.e, np.e**2])
        result = tf_exp_inv(x)
        expected = np.log(x)
        np.testing.assert_allclose(result, expected)

    def test_tf_square(self):
        """tf_square should square elements."""
        x = np.array([1.0, 2.0, 3.0])
        result = tf_square(x)
        expected = np.array([1.0, 4.0, 9.0])
        np.testing.assert_array_equal(result, expected)

    def test_tf_square_inv(self):
        """tf_square_inv should take square root."""
        x = np.array([1.0, 4.0, 9.0])
        result = tf_square_inv(x)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, expected)

    def test_tf_identity(self):
        """tf_identity should return input unchanged."""
        x = np.array([1.0, 2.0, 3.0])
        result = tf_identity(x)
        np.testing.assert_array_equal(result, x)

    def test_tf_explinear(self):
        """tf_explinear should apply explinear transform."""
        x = np.array([0.0, 1.0, 2.0])
        result = tf_explinear(x)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result)))
        
    def test_tf_explinear_inv(self):
        """tf_explinear_inv should apply inverse explinear transform."""
        x = np.array([0.5, 1.0, 2.0])
        result = tf_explinear_inv(x)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_tf_explinear_inv_large(self):
        """tf_explinear_inv should handle large values correctly."""
        x = np.array([35.0, 100.0])
        result = tf_explinear_inv(x)
        # For large x, should return approximately x
        np.testing.assert_allclose(result[1], x[1], rtol=1e-10)


class TestUnivRepr(unittest.TestCase):
    """Tests for _univ_repr utility function."""

    def test_univ_repr_scalar_values(self):
        """_univ_repr should format scalar values nicely."""
        class MockObject:
            pass
        obj = MockObject()
        obj.n_samples = 128
        obj.dimension = 5
        result = _univ_repr(obj, "MockClass", ["n_samples", "dimension"])
        self.assertIn("MockClass", result)
        self.assertIn("n_samples", result)
        self.assertIn("dimension", result)

    def test_univ_repr_power_of_two(self):
        """_univ_repr should recognize powers of 2."""
        class MockObject:
            pass
        obj = MockObject()
        obj.n = 512  # 2^9
        result = _univ_repr(obj, "Test", ["n"])
        self.assertIn("2^(9)", result)

    def test_univ_repr_array_values(self):
        """_univ_repr should handle numpy arrays."""
        class MockObject:
            pass
        obj = MockObject()
        obj.data = np.array([1.0, 2.0, 3.0])
        result = _univ_repr(obj, "Test", ["data"])
        self.assertIn("Test", result)

    def test_univ_repr_list_values(self):
        """_univ_repr should handle list values."""
        class MockObject:
            pass
        obj = MockObject()
        obj.seeds = [1, 2, 3]
        result = _univ_repr(obj, "Test", ["seeds"])
        self.assertIn("Test", result)


class TestData(unittest.TestCase):
    """Tests for Data class."""

    def test_data_init(self):
        """Data should initialize with parameters list."""
        d = Data(["param1", "param2"])
        self.assertEqual(d.parameters, ["param1", "param2"])

    def test_data_repr_empty(self):
        """Data.__repr__ should generate proper string with empty params."""
        d = Data([])
        d.time_integrate = 0.0
        result = str(d)
        self.assertIn("Data", result)

    def test_data_repr_with_values(self):
        """Data.__repr__ should show attributes when parameters are set."""
        d = Data(["test_param"])
        d.test_param = 42
        d.time_integrate = 0.5
        result = str(d)
        self.assertIn("Data", result)
        self.assertIn("test_param", result)

    def test_data_repr_with_stopping_crit(self):
        """Data.__repr__ should show stopping criterion if set."""
        d = Data([])
        d.time_integrate = 0.0
        d.stopping_crit = "test_crit"
        result = str(d)
        self.assertIn("Data", result)
        self.assertIn("test_crit", result)

    def test_data_repr_with_integrand(self):
        """Data.__repr__ should show integrand if set."""
        d = Data([])
        d.time_integrate = 0.0
        d.integrand = "test_integrand"
        result = str(d)
        self.assertIn("test_integrand", result)

    def test_data_repr_with_true_measure(self):
        """Data.__repr__ should show true_measure if set."""
        d = Data([])
        d.time_integrate = 0.0
        d.true_measure = "test_measure"
        result = str(d)
        self.assertIn("test_measure", result)

    def test_data_repr_with_discrete_distrib(self):
        """Data.__repr__ should show discrete_distrib if set."""
        d = Data([])
        d.time_integrate = 0.0
        d.discrete_distrib = "test_distrib"
        result = str(d)
        self.assertIn("test_distrib", result)


class TestExceptionsWarnings(unittest.TestCase):
    """Tests for exception and warning classes."""

    def test_parameter_error_instantiation(self):
        """ParameterError should be instantiable with message."""
        err = ParameterError("Test error message")
        self.assertIn("Test error message", str(err))

    def test_parameter_error_inheritance(self):
        """ParameterError should inherit from Exception."""
        err = ParameterError("Test")
        self.assertIsInstance(err, Exception)

    def test_parameter_error_raised(self):
        """ParameterError should be raised and caught properly."""
        with self.assertRaises(ParameterError):
            raise ParameterError("test")

    def test_method_implementation_error_instantiation(self):
        """MethodImplementationError needs subclass and method_name."""
        class TestClass:
            pass
        err = MethodImplementationError(TestClass(), "test_method")
        self.assertIn("TestClass", str(err))
        self.assertIn("test_method", str(err))

    def test_method_implementation_error_inheritance(self):
        """MethodImplementationError should inherit from Exception."""
        class TestClass:
            pass
        err = MethodImplementationError(TestClass(), "test")
        self.assertIsInstance(err, Exception)

    def test_method_implementation_error_raised(self):
        """MethodImplementationError should be raised and caught."""
        class TestClass:
            pass
        with self.assertRaises(MethodImplementationError):
            raise MethodImplementationError(TestClass(), "test")


if __name__ == "__main__":
    unittest.main()
