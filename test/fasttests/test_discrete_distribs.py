from qmcpy import *
from qmcpy.util import *
from qmcpy.discrete_distribution.c_lib import c_lib
import os
import unittest
import ctypes
from numpy import *
import time

class TestDiscreteDistribution(unittest.TestCase):

    def test_size_unisgned_long(self):
        get_unsigned_long_size_cf = c_lib.get_unsigned_long_size
        get_unsigned_long_size_cf.argtypes = []
        get_unsigned_long_size_cf.restype = ctypes.c_uint8
        if os.name == 'nt':
            self.assertEqual(get_unsigned_long_size_cf(),4)
        else:
            self.assertEqual(get_unsigned_long_size_cf(),8)

    def test_size_unisgned_long_long(self):
        get_unsigned_long_long_size_cf = c_lib.get_unsigned_long_long_size
        get_unsigned_long_long_size_cf.argtypes = []
        get_unsigned_long_long_size_cf.restype = ctypes.c_uint8
        self.assertEqual(get_unsigned_long_long_size_cf(),8)

    def test_abstract_methods(self):
        for d in [3,[1,3,5]]:
            dds = [
                IIDStdUniform(d),
                Lattice(d,order='natural'),
                Lattice(d,order='mps'),
                Lattice(d,order='linear'),
                Lattice(d,generating_vector='lattice_vec_lnb.750.24.npy'),
                DigitalNetB2(d,randomize='LMS_DS',graycode=False),
                DigitalNetB2(d,randomize='DS'),
                DigitalNetB2(d,graycode=True),
                DigitalNetB2(d,generating_matrices='niederreiter_mat.1377.52.52.lsb.npy'),
                Halton(d,randomize='QRNG',generalize=True),
                Halton(d,randomize='QRNG',generalize=False),
                Halton(d,randomize='Owen',generalize=False),
            ]
            for dd in dds:
                for _dd in [dd]+dd.spawn(1):
                    x = _dd.gen_samples(4)
                    if _dd.mimics=='StdUniform':
                        self.assertTrue((x>0).all() and (x<1).all())
                    pdf = _dd.pdf(_dd.gen_samples(4))
                    self.assertTrue(pdf.shape==(4,))
                    self.assertTrue(x.shape==(4,3))
                    self.assertTrue(x.dtype==float64)
                    s = str(_dd)
    
    def test_spawn(self):
        d = 3
        for dd in [IIDStdUniform(d),Lattice(d),DigitalNetB2(d),Halton(d)]:
            s = 3
            for spawn_dim in [4,[1,4,6]]:
                spawns = dd.spawn(s=s,dimensions=spawn_dim)
                self.assertTrue(len(spawns)==s)
                self.assertTrue(all(type(spawn)==type(dd) for spawn in spawns))
                self.assertTrue((array([spawn.d for spawn in spawns])==spawn_dim).all())

        
class TestLattice(unittest.TestCase):
    """ Unit tests for Lattice DiscreteDistribution. """

    def test_gen_samples(self):
        for order in ['natural','mps']: # ['natural','mps','linear']
            lattice0123 = Lattice(dimension=4,order=order,randomize=False)
            x0123 = lattice0123.gen_samples(8,warn=False)
            lattice13 = Lattice(dimension=[1,3],order=order,randomize=False)
            x13 = lattice13.gen_samples(n_min=5,n_max=7)
            self.assertTrue((x0123[5:7,[1,3]]==x13).all())

    def test_linear_order(self):
        true_sample = array([
            [1. / 8, 3. / 8, 3. / 8, 1. / 8],
            [3. / 8, 1. / 8, 1. / 8, 3. / 8],
            [5. / 8, 7. / 8, 7. / 8, 5. / 8],
            [7. / 8, 5. / 8, 5. / 8, 7. / 8]])
        cpu_time = {}
        for is_parallel in [False, True]:
            start_time = time.process_time()
            distribution = Lattice(dimension=4, randomize=False, order='linear', is_parallel=is_parallel)
            ld_sample = distribution.gen_samples(n_min=4, n_max=8, warn=False)
            end_time = time.process_time()
            self.assertTrue((ld_sample==true_sample).all())
            cpu_time[is_parallel] = end_time - start_time

        self.assertTrue(cpu_time[True] <= cpu_time[False])

    def test_gail_order(self):
        true_sample = array([
            [1. / 8, 3. / 8, 3. / 8, 1. / 8],
            [5. / 8, 7. / 8, 7. / 8, 5. / 8],
            [3. / 8, 1. / 8, 1. / 8, 3. / 8],
            [7. / 8, 5. / 8, 5. / 8, 7. / 8]])
        for is_parallel in [False, True]:
            distribution = Lattice(dimension=4, randomize=False, order='natural', is_parallel=is_parallel)
            ld_sample = distribution.gen_samples(n_min=4, n_max=8)
            self.assertTrue((ld_sample==true_sample).all())


    def test_mps_order(self):
        true_sample = array([
            [1. / 8, 3. / 8, 3. / 8, 1. / 8],
            [3. / 8, 1. / 8, 1. / 8, 3. / 8],
            [5. / 8, 7. / 8, 7. / 8, 5. / 8],
            [7. / 8, 5. / 8, 5. / 8, 7. / 8]])
        for is_parallel in [False, True]:
            distribution = Lattice(dimension=4, randomize=False, order='mps', is_parallel=is_parallel)
            ld_sample = distribution.gen_samples(n_min=4,n_max=8)
            self.assertTrue((ld_sample==true_sample).all())

    def test_linear_order_not_power_of_2(self):
        l = Lattice(dimension=3, order='linear')
        x = l.gen_samples(n_min=2, n_max=5)
        assert x.shape==(4,3)

    def test_integer_generating_vectors(self):
        distribution = Lattice(dimension=4, generating_vector=27, randomize=False,seed=136)
        true_sample =array([
            [0.125, 0.875, 0.625, 0.375],
            [0.625, 0.375, 0.125, 0.875],
            [0.375, 0.625, 0.875, 0.125],
            [0.875, 0.125, 0.375, 0.625]])
        self.assertTrue((distribution.gen_samples(n_min=4,n_max=8)==true_sample).all())

class TestDigitalNetB2(unittest.TestCase):
    """ Unit tests for DigitalNetB2 DiscreteDistribution. """

    def test_mimics(self):
        distribution = Sobol(dimension=3, randomize=True)
        self.assertEqual(distribution.mimics, "StdUniform")

    def test_gen_samples(self):
        dn123 = DigitalNetB2(dimension=4,graycode=False,randomize=False)
        x0123 = dn123.gen_samples(8)
        dn13 = DigitalNetB2(dimension=[1,3],graycode=False,randomize=False)
        x13 = dn13.gen_samples(n_min=4,n_max=8)
        self.assertTrue((x0123[4:8,[1,3]]==x13).all())
        dn123 = DigitalNetB2(dimension=4,graycode=True,randomize=False)
        x0123 = dn123.gen_samples(8)
        dn13 = DigitalNetB2(dimension=[1,3],graycode=True,randomize=False)
        x13 = dn13.gen_samples(n_min=5,n_max=7)
        self.assertTrue((x0123[5:7,[1,3]]==x13).all())
    
    def test_graycode_ordering(self):
        dnb2 = DigitalNetB2(2,randomize=False,graycode=True)
        x = dnb2.gen_samples(n_min=4,n_max=8)
        x_true = array([
            [ 0.375,  0.375],
            [ 0.875,  0.875],
            [ 0.625,  0.125],
            [ 0.125,  0.625]])
        self.assertTrue((x==x_true).all())

    def test_natural_ordering(self):
        dnb2 = DigitalNetB2(2,randomize=False,graycode=False)
        x = dnb2.gen_samples(n_min=4,n_max=8)
        x_true = array([
            [ 0.125,  0.625],
            [ 0.625,  0.125],
            [ 0.375,  0.375],
            [ 0.875,  0.875]])
        self.assertTrue((x==x_true).all())
    

class TestHalton(unittest.TestCase):
    """ Unit test for Halton DiscreteDistribution. """
    
    def test_gen_samples(self):
        h123 = Halton(dimension=4,randomize=False)
        x0123 = h123.gen_samples(8,warn=False)
        h13 = Halton(dimension=[1,3],randomize=False)
        x13 = h13.gen_samples(n_min=5,n_max=7,warn=False)
        self.assertTrue((x0123[5:7,[1,3]]==x13).all())
    
    def test_unrandomized(self):
        x_ur = Halton(dimension=2, randomize=False).gen_samples(4,warn=False)
        x_true = array([
            [0,     0],
            [1./2,  1./3],
            [1./4,  2./3],
            [3./4,  1./9]])
        self.assertTrue((x_ur==x_true).all())
    

if __name__ == "__main__":
    unittest.main()