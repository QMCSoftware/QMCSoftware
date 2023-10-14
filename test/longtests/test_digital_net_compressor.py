""" Unit tests for DigitalNetDataCompressor in QMCPy """

import unittest
import numpy as np
import qmcpy as qp
import os


class DigitalNetDataCompressorTest(unittest.TestCase):
    
    def setUp(self):
        self.path = os.path.dirname(__file__) + os.sep

    def test_compute_weights(self):
        x = np.loadtxt(f"{self.path}test_data/reg_x.csv", delimiter=',')
        y = np.loadtxt(f"{self.path}test_data/reg_y.csv", delimiter=",")
        weights_true = np.loadtxt(f"{self.path}test_data/reg_weights.csv", delimiter=",")
        sobol = np.loadtxt(f"{self.path}test_data/sobol.dat")
        dn = qp.DigitalNetDataCompressor(nu=1, m=6, dataset=x, labels=y, alpha=1, sobol=sobol)
        dn.compute_weights()
        weights, z = dn.weights, dn.sobol
        self.assertTrue(np.allclose(weights, weights_true, atol=1e-3))

    def test_approx_mean_mxy(self):
        x = np.loadtxt(f"{self.path}test_data/reg_x.csv", delimiter=',')
        y = np.loadtxt(f"{self.path}test_data/reg_y.csv", delimiter=",")
        z_true = np.loadtxt(f"{self.path}test_data/reg_z.csv", delimiter=",")
        weights_true = np.loadtxt(f"{self.path}test_data/reg_weights.csv", delimiter=",")
        sobol = np.loadtxt(f"{self.path}test_data/sobol.dat")
        dn = qp.DigitalNetDataCompressor(nu=1, m=6, dataset=x, labels=y, alpha=1, sobol=sobol)
        dn.approx_mean_mxy()
        weights, z = dn.weights, dn.sobol
        self.assertTrue(np.allclose(z, z_true, atol=1e-3))
        self.assertTrue(np.allclose(weights, weights_true, atol=1e-3))