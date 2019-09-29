#!/usr/bin/env python
'''
To install qmcpy, open a terminal, in directory QMCSoftware/qmcpy, run:

    python setup.py install

To uninstall, do the following:

    pip uninstall qmcpy

Reference:  Writing the Setup Script, https://docs.python.org/3/distutils/setupscript.html
'''
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="qmcpy",
    version="0.1",
    author="Fred Hickernell, Aleksei Sorokin, and Sou-Cheng T. Choi",
    author_email="hickernell@iit.com",
    description="(Quasi) Monte Carlo Framework in Python 3.0",
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
)