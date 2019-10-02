#!/usr/bin/env python
'''
To install qmcpy, open a terminal, in directory QMCSoftware/qmcpy, run:

    python setup.py install
    python setup.py clean

To uninstall, do the following:

    pip uninstall qmcpy

Reference:  Writing the Setup Script, https://docs.python.org/3/distutils/setupscript.html
'''
import setuptools

import os
from setuptools import setup, Command
class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./qmcpy.egg-info')

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qmcpy",
    version="0.1",
    author="Fred Hickernell, Aleksei Sorokin, and Sou-Cheng T. Choi",
    author_email="hickernell@iit.com",
    description="(Quasi) Monte Carlo Framework in Python 3.7",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QMCSoftware/QMCSoftware",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    cmdclass={
        'clean': CleanCommand,
    }
)

