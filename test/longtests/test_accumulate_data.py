import unittest
import os
from qmcpy.accumulate_data._accumulate_data import AccumulateData
from qmcpy.accumulate_data.ld_transform_data import LDTransformData
import numpy as np

def dummy_coefv(nl):
    return np.exp(-2*np.pi*1j*np.arange(nl)/(2*nl))

def dummy_fudge(m):
    return 5.*2.**(-m)

class TestAccumulateDataSerialization(unittest.TestCase):
    def test_save_and_load(self):
        # Create a minimal LDTransformData object
        obj = LDTransformData(
            m_min=4,
            m_max=6,
            coefv=dummy_coefv,
            fudge=dummy_fudge,
            check_cone=False,
            ncv=0,
            cv_mu=0,
            update_beta=False
        )
        # Set some attributes
        obj.y_val = np.array([1.0, 2.0, 3.0])
        obj.muhat = 1.23
        # Save to disk
        path = 'test_accumulatedata.pkl'
        obj.save(path)
        self.assertTrue(os.path.exists(path))
        # Load from disk
        loaded = AccumulateData.load(path)
        self.assertTrue(np.allclose(loaded.y_val, obj.y_val))
        self.assertEqual(loaded.muhat, obj.muhat)
        # Clean up
        os.remove(path)

    def test_compression_reduces_file_size(self):
        """Test that compress=True produces smaller files than compress=False"""
        # Create a minimal LDTransformData object with some data
        obj = LDTransformData(
            m_min=4,
            m_max=6,
            coefv=dummy_coefv,
            fudge=dummy_fudge,
            check_cone=False,
            ncv=0,
            cv_mu=0,
            update_beta=False
        )
        # Add some data to make compression meaningful
        obj.y_val = np.random.rand(1000)  # Larger array for better compression
        obj.muhat = 1.23
        obj.sighat = 0.45
        
        # Save without compression
        path_uncompressed = 'test_uncompressed.pkl'
        obj.save(path_uncompressed, compress=False)
        
        # Save with compression (note: .gz will be auto-appended)
        path_compressed_base = 'test_compressed.pkl'
        obj.save(path_compressed_base, compress=True)
        path_compressed = path_compressed_base + '.gz'  # The actual file created
        
        # Check that both files exist
        self.assertTrue(os.path.exists(path_uncompressed))
        self.assertTrue(os.path.exists(path_compressed))
        
        # Get file sizes
        size_uncompressed = os.path.getsize(path_uncompressed)
        size_compressed = os.path.getsize(path_compressed)
        
        # Assert that compressed file is smaller
        self.assertLess(size_compressed, size_uncompressed, 
                       f"Compressed file ({size_compressed} bytes) should be smaller than "
                       f"uncompressed file ({size_uncompressed} bytes)")
        
        # Verify both files load correctly with the same data
        loaded_uncompressed = AccumulateData.load(path_uncompressed, compressed=False)
        loaded_compressed = AccumulateData.load(path_compressed)
        
        # Assert compressed file can be read back with correct original values
        self.assertTrue(np.allclose(loaded_compressed.y_val, obj.y_val),
                       "Compressed file should contain the original y_val data")
        self.assertEqual(loaded_compressed.muhat, obj.muhat,
                        "Compressed file should contain the original muhat value")
        self.assertEqual(loaded_compressed.sighat, obj.sighat,
                        "Compressed file should contain the original sighat value")
        
        # Verify both compressed and uncompressed files contain identical data
        self.assertTrue(np.allclose(loaded_uncompressed.y_val, loaded_compressed.y_val),
                       "Compressed and uncompressed files should contain identical y_val data")
        self.assertEqual(loaded_uncompressed.muhat, loaded_compressed.muhat,
                        "Compressed and uncompressed files should contain identical muhat values")
        
        # Clean up
        os.remove(path_uncompressed)
        os.remove(path_compressed)

    def test_compression_auto_appends_gz_extension(self):
        """Test that compress=True automatically appends .gz extension"""
        # Create test data
        obj = LDTransformData(
            m_min=4,
            m_max=6,
            coefv=dummy_coefv,
            fudge=dummy_fudge,
            check_cone=False,
            ncv=0,
            cv_mu=0,
            update_beta=False
        )
        obj.y_val = np.array([1.0, 2.0, 3.0])
        obj.muhat = 1.23
        
        # Test auto-appending .gz
        base_path = 'test_auto_gz.pkl'
        expected_path = base_path + '.gz'
        
        obj.save(base_path, compress=True)
        
        # .gz version should exist
        self.assertTrue(os.path.exists(expected_path),
                       "File with .gz extension should exist when compress=True")
        
        # Load with auto-detection
        loaded = AccumulateData.load(expected_path)  # No need to specify compressed=True
        self.assertEqual(loaded.muhat, obj.muhat)
        
        # Test that .gz is not double-appended
        already_gz_path = 'test_already.pkl.gz'
        obj.save(already_gz_path, compress=True)
        
        self.assertTrue(os.path.exists(already_gz_path))
        self.assertFalse(os.path.exists(already_gz_path + '.gz'),
                        "Should not double-append .gz extension")
        
        # Clean up
        os.remove(expected_path)
        os.remove(already_gz_path)

if __name__ == '__main__':
    unittest.main()
