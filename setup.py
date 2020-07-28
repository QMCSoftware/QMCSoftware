import setuptools
from setuptools import Extension
from setuptools.command.install import install
from setuptools import Command
import os
import subprocess

class CustomInstall(install):
    """Custom handler for the 'install' command."""
    def run(self):
        try:
            subprocess.check_call('make qrng',shell=True)
        except:
            print('Problem compiling qrng c files')
        super().run()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./qmcpy/qmcpy.egg-info")

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

packages = [
    'qmcpy',
    'qmcpy.true_measure',
    'qmcpy.stopping_criterion',
    'qmcpy.discrete_distribution',
    'qmcpy.accumulate_data',
    'qmcpy.util',
    'qmcpy.integrand',
    'qmcpy.discrete_distribution.lattice',
    'qmcpy.discrete_distribution.qrng',
    'qmcpy.discrete_distribution.sobol']

setuptools.setup(
    name="qmcpy",
    version="0.2",
    author="Fred Hickernell, Sou-Cheng T. Choi, Mike McCourt, Jagadeeswaran Rathinavel, Aleksei Sorokin",
    author_email="asorokin@hawk.iit.edu",
    license='Apache license 2.0',
    description="(Quasi) Monte Carlo Framework in Python 3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://qmcsoftware.github.io/QMCSoftware/",
    download_url="https://github.com/QMCSoftware/QMCSoftware/archive/v0.1-alpha.tar.gz",
    packages=packages,
    install_requires=[
        'scipy >= 1.2.0',
        'numpy >= 1.18.5'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"],
    keywords=['quasi','monte','carlo','community','software','cubature','numerical','integration','discrepancy','sobol','lattice'],
    python_requires=">=3.5",
    ext_modules=[
        Extension(
            name='qmcpy.discrete_distribution.qrng.qrng_lib',
            sources=['qmcpy/discrete_distribution/qrng/ghalton.c',
                    'qmcpy/discrete_distribution/qrng/korobov.c',
                    'qmcpy/discrete_distribution/qrng/MRG63k3a.c',
                    'qmcpy/discrete_distribution/qrng/sobol.c'],
            extra_compile_args=['-fPIC','-shared','-lm'])],
     cmdclass={
        'clean': CleanCommand,
        'install': CustomInstall}
)
