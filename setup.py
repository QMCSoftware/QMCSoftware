'''
To install: open a terminal and in directory QMCSoftware, run:

    python setup.py install
    python setup.py clean

This will put qmcpy in  ~/anaconda3/envs/qmcpy/lib/python3.7/site-packages/

To uninstall, do the following:

    pip uninstall qmcpy
'''
import os

import setuptools
from setuptools import Command


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./qmcpy/qmcpy.egg-info")


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qmcpy",
    version="0.1",
    author="Fred Hickernell, Aleksei Sorokin, and Sou-Cheng T. Choi",
    author_email="hickernell@iit.com",
    description="(Quasi) Monte Carlo Framework in Python 3.6 and 3.7",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QMCSoftware/QMCSoftware",
    package_dir={"": "qmcpy"},
    packages=setuptools.find_packages("qmcpy"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: IIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    cmdclass={"clean": CleanCommand}
)
