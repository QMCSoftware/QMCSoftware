import unittest
import subprocess
import os

import sys, os

import workouts.wo_asian_option as wo_asian_option
import workouts.wo_keister as wo_keister
import workouts.wo_3d_point_distribution as wo_3d_point_distribution
import workouts.wo_abstol_runtime as wo_abstol_runtime

class TestWoAsianOption(unittest.TestCase):

    def test_output(self):
        wo_asian_option.test_distributions()


class TestWoKeister(unittest.TestCase):

    def test_output(self):
        wo_keister.test_distributions()


class TestAbsTolRunTime(unittest.TestCase):

    def test_output(self):
        import imp
        runpy = imp.load_source('__main__', './workouts/wo_abstol_runtime.py')

class TestWo3DPointDistribution(unittest.TestCase):
    def test_output(self):
        wo_3d_point_distribution.plot3d()



