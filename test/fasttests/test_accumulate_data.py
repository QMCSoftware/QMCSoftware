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

if __name__ == '__main__':
    unittest.main()
