import setuptools
from setuptools import Extension
from setuptools.command.install import install
from setuptools import Command
import os
import subprocess




class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./qmcpy/qmcpy.egg-info")

try:
    with open("README.md", "r", encoding="utf-8", errors='ignore') as fh:
        long_description = fh.read()
except:
    long_description = "QMCPy"

packages = [
    'qmcpy',
    'qmcpy.true_measure',
    'qmcpy.stopping_criterion',
    'qmcpy.discrete_distribution',
    'qmcpy.accumulate_data',
    'qmcpy.util',
    'qmcpy.integrand',
    'qmcpy.discrete_distribution.lattice',
    'qmcpy.discrete_distribution.c_lib',
    'qmcpy.discrete_distribution.sobol']

setuptools.setup(
    name="qmcpy",
    version="0.4",
    author="Fred Hickernell, Sou-Cheng T. Choi, Mike McCourt, Jagadeeswaran Rathinavel, Aleksei Sorokin",
    author_email="asorokin@hawk.iit.edu",
    license='Apache license 2.0',
    description="(Quasi) Monte Carlo Framework in Python 3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://qmcsoftware.github.io/QMCSoftware/",
    download_url="https://github.com/QMCSoftware/QMCSoftware/releases/tag/v0.4.gz",
    packages=packages,
    install_requires=[
        'scipy >= 1.2.0',
        'numpy >= 1.18.5'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"],
    keywords=['quasi', 'monte', 'carlo', 'community', 'software', 'cubature', 'numerical', 'integration', 'discrepancy',
              'sobol', 'lattice'],
    python_requires=">=3.5",
    include_package_data=True,
    ext_modules=[
        Extension(
            name='qmcpy.discrete_distribution.c_lib.c_lib',
            sources=[
                'qmcpy/discrete_distribution/c_lib/halton_owen.c',
                'qmcpy/discrete_distribution/c_lib/halton_qrng.c',
                'qmcpy/discrete_distribution/c_lib/korobov_qrng.c',
                'qmcpy/discrete_distribution/c_lib/sobol_qrng.c',
                'qmcpy/discrete_distribution/c_lib/sobol_seq51.c', 
                'qmcpy/discrete_distribution/c_lib/MRG63k3a.c'],
            extra_compile_args=['-shared'])],
    cmdclass={'clean': CleanCommand})
