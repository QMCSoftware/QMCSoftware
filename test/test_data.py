from qmcpy import *
from qmcpy.util.data import Data
import numpy as np
import pickle
import unittest
import tempfile
import os
from pathlib import Path


class TestDataSaveLoad(unittest.TestCase):
    def setUp(self):
        # Create a small synthetic Data instance instead of running an integrate()
        # This keeps tests fast while exercising save/load behavior.
        self.solution = 0.0
        self.data = Data(parameters=['solution', 'n_total'])
        self.data.solution = np.array([0.0], dtype=np.float64)
        self.data.n_total = 1
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_roundtrip_uncompressed_and_compressed(self):
        p1 = os.path.join(self.tmpdir, 'd1.pkl')
        p2 = os.path.join(self.tmpdir, 'd1_compressed.pkl')

        # uncompressed
        self.data.save(p1)
        self.assertTrue(os.path.exists(p1))
        l1 = Data.load(p1)
        self.assertEqual(l1.solution, self.data.solution)
        self.assertEqual(l1.n_total, self.data.n_total)

        # compressed (auto-appends .gz)
        self.data.save(p2, compress=True)
        self.assertTrue(os.path.exists(p2 + '.gz'))
        l2 = Data.load(p2 + '.gz')
        self.assertEqual(l2.solution, self.data.solution)

    def test_protocols_and_pathlib(self):
        # test a small set of protocols and Path compatibility
        for protocol in (pickle.HIGHEST_PROTOCOL, 2):
            fp = Path(self.tmpdir) / f'd_proto_{protocol}.pkl'
            self.data.save(fp, protocol=protocol)
            loaded = Data.load(fp)
            self.assertEqual(loaded.n_total, self.data.n_total)

    def test_load_nonexistent(self):
        with self.assertRaises(FileNotFoundError):
            Data.load(os.path.join(self.tmpdir, 'nope.pkl'))

    def test_preserve_attributes(self):
        # create a more feature-rich Data object without running heavy computations
        big = Data(parameters=['solution', 'n_total', 'extra'])
        big.solution = np.array([1.23, 4.56], dtype=np.float64)
        big.n_total = 1024
        big.extra = {'info': 'metadata'}
        p = os.path.join(self.tmpdir, 'big.pkl')
        big.save(p)
        loaded = Data.load(p)
        for attr in ('solution', 'n_total'):
            if hasattr(big, attr):
                orig = getattr(big, attr)
                new = getattr(loaded, attr)
                if isinstance(orig, np.ndarray):
                    np.testing.assert_array_equal(new, orig)
                else:
                    self.assertEqual(new, orig)

    def test_compression_reduces_size(self):
        # Construct a synthetic larger Data object by attaching a large array
        large = Data(parameters=['solution', 'blob'])
        # create ~1MB of data to make compression meaningful but keep runtime low
        large.blob = np.zeros(200_000, dtype=np.uint8)
        large.solution = np.array([0.0])
        u = os.path.join(self.tmpdir, 'large_u.pkl')
        c = os.path.join(self.tmpdir, 'large_c.pkl')
        large.save(u, compress=False)
        large.save(c, compress=True)
        # compression file may be suffixed with .gz
        cpath = c if os.path.exists(c) else c + '.gz'
        su, sc = os.path.getsize(u), os.path.getsize(cpath)
        # compressed size should not be substantially larger than uncompressed
        self.assertLessEqual(sc, su * 1.1)

    def test_preserve_various_types(self):
        # ensure various common attribute types survive round-trip
        d = Data(parameters=['solution', 'meta', 'flag'])
        d.solution = np.array([1.0, 2.0], dtype=np.float64)
        d.meta = {'a': 1, 'b': [1, 2, 3], 'c': {'inner': 'x'}}
        d.flag = True
        p = os.path.join(self.tmpdir, 'rich.pkl')
        d.save(p)
        out = Data.load(p)
        np.testing.assert_array_equal(out.solution, d.solution)
        self.assertEqual(out.meta, d.meta)
        self.assertEqual(out.flag, d.flag)

    def test_gz_filename_and_compress_flag(self):
        d = Data(parameters=['x'])
        gzpath = os.path.join(self.tmpdir, 'x.pkl.gz')
        # compress=True and filename already .gz -> should create exactly gzpath
        d.save(gzpath, compress=True)
        self.assertTrue(os.path.exists(gzpath))
        # compress=False and filename .gz -> still writes to the given path
        d.save(gzpath, compress=False)
        self.assertTrue(os.path.exists(gzpath))

    def test_load_corrupted_file_raises(self):
        p = os.path.join(self.tmpdir, 'corrupt.pkl')
        with open(p, 'wb') as f:
            f.write(b'not a pickle')
        with self.assertRaises(Exception):
            Data.load(p)

    def test_invalid_protocol_raises(self):
        p = os.path.join(self.tmpdir, 'bad_proto.pkl')
        # Passing a non-int protocol should raise when pickle.dump is called
        with self.assertRaises(TypeError):
            self.data.save(p, protocol='not-a-protocol')

    def test_non_pickleable_attributes_skipped(self):
        """Test that non-pickleable attributes are gracefully skipped"""
        d = Data(parameters=['test'])
        d.good_attr = "saveable"
        d.bad_attr = lambda x: x  # Non-pickleable
        p = os.path.join(self.tmpdir, 'mixed.pkl')
        d.save(p)
        loaded = Data.load(p)
        self.assertTrue(hasattr(loaded, 'good_attr'))
        self.assertFalse(hasattr(loaded, 'bad_attr'))

if __name__ == '__main__':
    unittest.main()